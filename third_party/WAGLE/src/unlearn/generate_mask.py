import os
from itertools import islice
from time import time

import torch
import tqdm
from torch.utils.data import SequentialSampler
from transformers import Trainer


def select_lowest_score_positions(scores, ratio):
    """Return WAGLE's exact-budget mask using its original argsort ordering."""
    selected_count = int(scores.numel() * ratio)
    selected = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    if selected_count:
        positions = torch.argsort(scores)
        selected[positions[:selected_count]] = True
        del positions
    return selected


def save_masks_from_scores(scores, ratios, mask_dir, parameter_layout):
    """Materialize WAGLE masks after callers have released scoring state."""
    if sum(num_elements for _, _, num_elements in parameter_layout) != len(scores):
        raise RuntimeError("WAGLE score vector does not match model parameter layout.")

    for ratio in ratios:
        output_path = os.path.join(mask_dir, f"with_{ratio}.pt")
        if os.path.exists(output_path):
            continue

        selected = select_lowest_score_positions(scores, ratio)
        hard_dict = {}
        start_index = 0
        for key, shape, num_elements in parameter_layout:
            hard_dict[key] = selected[start_index : start_index + num_elements].reshape(
                shape
            )
            start_index += num_elements
        torch.save(hard_dict, output_path)
        del hard_dict, selected


class GenerateMask(Trainer):
    def __init__(
        self,
        score_type,
        ratios,
        mask_dir,
        p,
        q,
        mu,
        max_score_batches=None,
        forget_score_batches=None,
        retain_score_batches=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.score_type = score_type
        self.ratios = [float(ratio) for ratio in ratios]
        if not self.ratios or any(not 0.0 < ratio <= 1.0 for ratio in self.ratios):
            raise ValueError("Mask ratios must be non-empty values in (0, 1].")
        self.max_score_batches = self._validate_score_batch_limit(
            "max_score_batches", max_score_batches
        )
        self.forget_score_batches = self._validate_score_batch_limit(
            "forget_score_batches",
            max_score_batches if forget_score_batches is None else forget_score_batches,
        )
        self.retain_score_batches = self._validate_score_batch_limit(
            "retain_score_batches",
            max_score_batches if retain_score_batches is None else retain_score_batches,
        )
        self.score_batch_counts = {}
        self.mask_dir = mask_dir
        self.p = p
        self.q = q
        self.mu = mu

    @staticmethod
    def _validate_score_batch_limit(name, value):
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive when provided.")
        return value

    def _get_train_sampler(self):
        if os.environ.get("FAIR_WMDP_SEQUENTIAL") == "1":
            if self.train_dataset is None:
                raise RuntimeError("Sequential fair scoring requires a train dataset.")
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()

    def snip_advanced(self, CL=False):
        begin_time = time()
        forget_gradint = {}
        retain_gradint = {}
        self.accelerator.free_memory()
        self.model.eval()
        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        forget_batches = 0
        forget_dataloader = self.get_train_dataloader()
        for inputs in tqdm.tqdm(
            islice(forget_dataloader, self.forget_score_batches),
            desc="computing forget gradient",
        ):
            forget_batches += 1
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "forget", CL=CL)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in forget_gradint:
                        forget_gradint[key] = tensor.grad.data
                    else:
                        forget_gradint[key] += tensor.grad.data

            model.zero_grad()
        if (
            self.forget_score_batches is not None
            and forget_batches != self.forget_score_batches
        ):
            raise ValueError(
                "forget_score_batches exceeds the available training stream."
            )
        self.score_batch_counts["forget"] = forget_batches
        retain_batches = 0
        retain_dataloader = self.get_train_dataloader()
        for inputs in tqdm.tqdm(
            islice(retain_dataloader, self.retain_score_batches),
            desc="computing retain gradient",
        ):
            retain_batches += 1
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "retain")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in retain_gradint:
                        retain_gradint[key] = tensor.grad.data
                    else:
                        retain_gradint[key] += tensor.grad.data
            model.zero_grad()
        if (
            self.retain_score_batches is not None
            and retain_batches != self.retain_score_batches
        ):
            raise ValueError(
                "retain_score_batches exceeds the available training stream."
            )
        self.score_batch_counts["retain"] = retain_batches

        with torch.no_grad():
            scores = {}
            for key, tensor in model.named_parameters():
                scores[key] = -torch.abs(
                    (tensor.data - retain_gradint[key] / self.mu) * forget_gradint[key]
                )
            end_time = time()
            print(f"Mask generation Time taken: {end_time-begin_time}")
            self.scores = torch.cat(
                [scores.flatten().cpu() for scores in scores.values()]
            )

    def compute_loss_adapted(
        self, model, inputs, key, CL=False, FT=False, return_outputs=False
    ):
        data = inputs[key]
        retain_data = inputs["retain"]
        if CL and key == "forget":
            forget_data = data
            input_ids = forget_data[0].clone()
            labels = forget_data[3]
            postions = forget_data[4]
            pad_id = input_ids[0][-1].item()
            for idx, position in enumerate(postions):
                input_ids[idx, position:] = labels[idx][position:].clone()
                mask = input_ids[idx] == -100
                input_ids[idx, mask] = pad_id
            inputs = {
                "input_ids": input_ids,
                "attention_mask": forget_data[1],
                "labels": labels,
            }
        else:
            inputs = {
                "input_ids": data[0],
                "attention_mask": data[1],
                "labels": data[2],
            }

        outputs = model(**inputs)

        loss = outputs.loss
        if FT:
            retain_inputs = {
                "input_ids": retain_data[0],
                "attention_mask": retain_data[1],
                "labels": retain_data[2],
            }
            retain_outputs = model(**retain_inputs)
            loss += retain_outputs.loss
        return (loss, outputs) if return_outputs else loss
