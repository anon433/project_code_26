# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Any, Dict, List, Optional, Union

import logging
import os

import torch
from torch.utils.data import Dataset, SequentialSampler
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)


class _CompleteEpochCallback(TrainerCallback):
    """Stop after complete data passes and flush the final partial accumulation."""

    def __init__(self, trainer, epoch_count, final_remainder):
        self.trainer = trainer
        self.epoch_count = int(epoch_count)
        self.final_remainder = int(final_remainder)
        self.epochs_completed = 0

    def on_train_begin(self, args, state, control, **kwargs):
        del args, kwargs
        state.num_train_epochs = self.epoch_count
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        del args, kwargs
        self.epochs_completed += 1
        if self.epochs_completed < self.epoch_count:
            return control
        if self.final_remainder:
            self.trainer._flush_partial_gradient_accumulation(self.final_remainder)
            control = self.trainer.control
        state.epoch = float(self.epoch_count)
        state.num_train_epochs = self.epoch_count
        control.should_training_stop = True
        return control


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        self.complete_epoch_flushes = 0
        super().__init__(*args, **kwargs)
        epoch_count = getattr(self.args, "_complete_epoch_count", None)
        if epoch_count is not None:
            self.add_callback(
                _CompleteEpochCallback(
                    self,
                    epoch_count=epoch_count,
                    final_remainder=self.args._complete_epoch_final_remainder,
                )
            )

    def _get_train_sampler(self):
        if os.environ.get("FAIR_WMDP_SEQUENTIAL") == "1":
            if self.train_dataset is None:
                raise RuntimeError("Strict fair-WMDP training requires a dataset.")
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()

    def _flush_partial_gradient_accumulation(self, remainder):
        if self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError(
                "Cannot flush partial accumulation before optimizer setup."
            )
        if not 0 < remainder < self.args.gradient_accumulation_steps:
            raise ValueError("Partial accumulation remainder is invalid.")
        scale = self.args.gradient_accumulation_steps / remainder
        gradients = []
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
                    gradients.append(parameter)
        if not gradients:
            raise RuntimeError("Final partial accumulation has no gradients.")

        self.accelerator.gradient_state._set_sync_gradients(True)
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                gradients,
                self.args.max_grad_norm,
            )
        self.control = self.callback_handler.on_pre_optimizer_step(
            self.args,
            self.state,
            self.control,
        )
        self.optimizer.step()
        self.control = self.callback_handler.on_optimizer_step(
            self.args,
            self.state,
            self.control,
        )
        if not self.accelerator.optimizer_step_was_skipped and not isinstance(
            self.lr_scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            self.lr_scheduler.step()
        self.model.zero_grad()
        self.state.global_step += 1
        self.control = self.callback_handler.on_step_end(
            self.args,
            self.state,
            self.control,
        )
        self.complete_epoch_flushes += 1

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators:
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_metrics = {}
                    for _, evaluator in self.evaluators.items():
                        eval_args = {
                            "output_dir": output_dir,
                            "template_args": self.template_args,
                            "model": self.model,
                            "tokenizer": self.tokenizer,
                        }
                        eval_metrics.update(evaluator.evaluate(**eval_args))
                    self.log(eval_metrics)
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
