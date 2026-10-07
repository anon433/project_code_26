"""Trainer registry for the paper's scoring and downstream objectives."""

import logging
import math
import os
from omegaconf import DictConfig, OmegaConf
from importlib import import_module

logger = logging.getLogger(__name__)
TRAINER_REGISTRY = {
    "FinetuneTrainer": "trainer.base",
    "GradDiff": "trainer.unlearn.grad_diff",
    "NPO": "trainer.unlearn.npo",
    "GradSaliencyMaskSearch": "trainer.unlearn.grad_saliency_mask",
    "TCUSTrajectoryGradDiff": "trainer.unlearn.tcus.trajectory",
    "TCUSTrajectoryNPO": "trainer.unlearn.tcus.trajectory",
}


def training_process_count() -> int:
    """Return optimizer replicas, not the number of model-parallel GPUs."""
    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except ValueError as error:
        raise ValueError(f"Invalid WORLD_SIZE={raw_world_size!r}.") from error
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}.")
    return world_size


def load_trainer_args(trainer_args: DictConfig, dataset):
    from transformers import TrainingArguments

    trainer_args = (
        OmegaConf.to_container(trainer_args, resolve=True)
        if OmegaConf.is_config(trainer_args)
        else dict(trainer_args)
    )
    warmup_epochs = trainer_args.pop("warmup_epochs", None)
    complete_epochs = trainer_args.pop("complete_epochs", False)
    complete_epoch_metadata = None
    if complete_epochs:
        if complete_epochs is not True:
            raise TypeError("complete_epochs must be a boolean.")
        if trainer_args.get("max_steps", -1) > 0:
            raise ValueError("complete_epochs cannot be combined with max_steps.")
        num_train_epochs = float(trainer_args["num_train_epochs"])
        if not num_train_epochs.is_integer() or num_train_epochs <= 0:
            raise ValueError(
                "complete_epochs requires a positive integer num_train_epochs."
            )
        batch_size = int(trainer_args["per_device_train_batch_size"])
        grad_accum_steps = int(trainer_args["gradient_accumulation_steps"])
        process_count = training_process_count()
        dataset_len = len(dataset)
        samples_per_microbatch = batch_size * process_count
        if trainer_args.get("dataloader_drop_last", False):
            microbatches_per_epoch = dataset_len // samples_per_microbatch
        else:
            microbatches_per_epoch = math.ceil(dataset_len / samples_per_microbatch)
        if microbatches_per_epoch <= 0:
            raise ValueError("complete_epochs requires a non-empty train dataset.")
        epoch_count = int(num_train_epochs)
        total_microbatches = epoch_count * microbatches_per_epoch
        trainer_args["max_steps"] = math.ceil(total_microbatches / grad_accum_steps)
        complete_epoch_metadata = {
            "epoch_count": epoch_count,
            "microbatches": total_microbatches,
            "final_remainder": total_microbatches % grad_accum_steps,
        }
    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        process_count = training_process_count()
        dataset_len = len(dataset)
        raw_warmup_steps = (warmup_epochs * dataset_len) / (
            batch_size * grad_accum_steps * process_count
        )
        trainer_args["warmup_steps"] = (
            math.ceil(raw_warmup_steps) if complete_epochs else int(raw_warmup_steps)
        )

    trainer_args = TrainingArguments(**trainer_args)
    if complete_epoch_metadata is not None:
        trainer_args._complete_epoch_count = complete_epoch_metadata["epoch_count"]
        trainer_args._complete_epoch_microbatches = complete_epoch_metadata[
            "microbatches"
        ]
        trainer_args._complete_epoch_final_remainder = complete_epoch_metadata[
            "final_remainder"
        ]
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    evaluators=None,
    template_args=None,
):
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get("method_args", {})
    trainer_args = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = trainer_cfg.get("handler")
    assert trainer_handler_name is not None, ValueError(
        f"{trainer_handler_name} handler not set"
    )
    module = TRAINER_REGISTRY.get(trainer_handler_name)
    trainer_cls = (
        getattr(import_module(module), trainer_handler_name) if module else None
    )
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_handler_name} not implemented or not registered"
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        **method_args,
    )
    logger.info(
        f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}"
    )
    return trainer, trainer_args
