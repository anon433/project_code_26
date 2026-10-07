"""Stage 1: Gradient-saliency mask search.

Computes per-parameter saliency scores using one of:
    "ratio":     score = F / (R + eps)
    "norm_diff": score = (F - R) / (F + R + eps)

where F = |dL_forget/dtheta_i|^2, R = |dL_retain/dtheta_i|^2.

Parameters with high forget-gradient and low retain-gradient are most
amenable to gradient-based unlearning in Stage 2. Saves continuous scores
for the runner's exact-budget mask selection.
"""

import json
import logging
import os
import re
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from transformers.trainer_utils import TrainOutput

from score_artifacts import save_selector_logits
from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


class GradSaliencyMaskSearch(UnlearnTrainer):
    SCORE_FNS = ("ratio", "norm_diff")

    def __init__(
        self,
        select_frac=0.10,
        ema_alpha=0.1,
        eps=1e-8,
        param_filter="all",
        score_fn="ratio",
        save_selector_logits=True,
        selector_logits_storage_dtype="float32",
        save_boolean_active_mask=False,
        independent_side_passes=False,
        forget_score_batches=None,
        retain_score_batches=None,
        consumption_manifest_path=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            select_frac: Fraction of params to select (top-k by saliency).
            ema_alpha: EMA smoothing factor for gradient accumulation.
            eps: Denominator stabiliser to avoid division by zero.
            param_filter: Which params to score. "all", "linear_weight",
                          or a regex matched against parameter names.
            score_fn: Scoring formula. "ratio" = F/(R+eps),
                      "norm_diff" = (F-R)/(F+R+eps).
        """
        super().__init__(*args, **kwargs)
        if score_fn not in self.SCORE_FNS:
            raise ValueError(
                f"Unknown score_fn={score_fn!r}, expected one of {self.SCORE_FNS}"
            )
        self.select_frac = select_frac
        self.ema_alpha = ema_alpha
        self.eps = eps
        self.param_filter = param_filter
        self.score_fn = score_fn
        self.save_selector_logits = save_selector_logits
        if selector_logits_storage_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "selector_logits_storage_dtype must be float16, bfloat16, or float32."
            )
        self.selector_logits_storage_dtype = selector_logits_storage_dtype
        self.save_boolean_active_mask = save_boolean_active_mask
        self.independent_side_passes = independent_side_passes
        self.forget_score_batches = forget_score_batches
        self.retain_score_batches = retain_score_batches
        self.consumption_manifest_path = consumption_manifest_path

        base_model = self._unwrap(self.model)

        # Build scored-parameter list and float32 accumulators.
        # Keep accumulators beside parameters during scoring to avoid copying
        # every 7B gradient tensor to CPU twice per batch. They are moved to
        # CPU only once when final selector artifacts are materialized.
        self._scored_params = []  # [(safe_name, original_name)]
        self.forget_acc = {}
        self.retain_acc = {}
        self.total_params = 0

        for name, p in base_model.named_parameters():
            if not self._should_mask(name, p):
                p.requires_grad = False
                continue
            safe = name.replace(".", "__")
            self._scored_params.append((safe, name))
            self.forget_acc[safe] = torch.zeros(
                p.shape, dtype=torch.float32, device=p.device
            )
            self.retain_acc[safe] = torch.zeros(
                p.shape, dtype=torch.float32, device=p.device
            )
            self.total_params += p.numel()

        logger.info(
            f"GradSaliencyMaskSearch: {self.total_params:,} params to score, "
            f"select_frac={select_frac}, ema_alpha={ema_alpha}, "
            f"score_fn={score_fn}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def _should_mask(self, name, param):
        if self.param_filter == "all":
            return True
        if self.param_filter == "linear_weight":
            return "weight" in name and param.dim() >= 2
        return bool(re.search(self.param_filter, name))

    def _compute_scores(self, safe):
        """Compute saliency score for parameter ``safe`` using self.score_fn."""
        F = self.forget_acc[safe]
        R = self.retain_acc[safe]
        if self.score_fn == "ratio":
            return F / (R + self.eps)
        else:  # norm_diff
            return (F - R) / (F + R + self.eps)

    def _get_train_sampler(self):
        if self.independent_side_passes:
            if self.train_dataset is None:
                raise RuntimeError("Independent side passes require a train dataset.")
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()

    # ------------------------------------------------------------------
    # Core: compute_loss — accumulates gradient statistics
    # ------------------------------------------------------------------

    def _accumulate_grads(self, acc_dict):
        """Read .grad from scored params and EMA-accumulate squared grads."""
        base_model = self._unwrap(self.model)
        alpha = self.ema_alpha
        for safe, orig in self._scored_params:
            p = base_model.get_parameter(orig)
            if p.grad is not None:
                gf = p.grad.float()
                acc_dict[safe].mul_(1 - alpha).addcmul_(gf, gf, value=alpha)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = {
            k: inputs["forget"][k] for k in ("input_ids", "attention_mask", "labels")
        }
        retain_inputs = {
            k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")
        }

        # --- Forget pass: forward + backward + accumulate ---
        forget_outputs = model(**forget_inputs)
        forget_loss = forget_outputs.loss
        forget_loss.backward()
        self._accumulate_grads(self.forget_acc)
        model.zero_grad()

        # --- Retain pass: forward + backward + accumulate ---
        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss
        retain_loss.backward()
        self._accumulate_grads(self.retain_acc)
        model.zero_grad()

        # Periodic logging
        if self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "saliency/forget_loss": forget_loss.item(),
                    "saliency/retain_loss": retain_loss.item(),
                }
            )

        # Return a dummy loss: the real work is in the accumulators.
        # The trainer calls backward on this, which is a no-op (leaf tensor).
        dummy = torch.tensor(0.0, device=forget_loss.device, requires_grad=True)
        return (dummy, forget_outputs) if return_outputs else dummy

    # ------------------------------------------------------------------
    # Independent side passes — sequential, no optimizer updates
    # ------------------------------------------------------------------

    def _accumulate_side_batch(self, model, inputs, side):
        selected = {
            key: inputs[side][key] for key in ("input_ids", "attention_mask", "labels")
        }
        outputs = model(**selected)
        outputs.loss.backward()
        self._accumulate_grads(self.forget_acc if side == "forget" else self.retain_acc)
        model.zero_grad()
        return float(outputs.loss.detach())

    def _resolved_side_batch_counts(self, loader):
        try:
            available_batches = len(loader)
        except TypeError as error:
            raise TypeError(
                "Independent side passes require a dataloader with a known length."
            ) from error

        requested = {
            "forget": self.forget_score_batches,
            "retain": self.retain_score_batches,
        }
        resolved = {}
        for side, count in requested.items():
            name = f"{side}_score_batches"
            if count is None:
                count = available_batches
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{name} must be a positive integer or None.")
            if count > available_batches:
                raise ValueError(
                    f"{name}={count} exceeds dataloader length {available_batches}."
                )
            resolved[side] = count
        return resolved

    def _run_independent_side_passes(self):
        if self.args.per_device_train_batch_size != 1:
            raise ValueError(
                "Independent side passes require per_device_train_batch_size == 1."
            )
        loader = self.get_train_dataloader()
        counts = self._resolved_side_batch_counts(loader)
        trace = []
        for side in ("forget", "retain"):
            for row, inputs in enumerate(loader):
                if row >= counts[side]:
                    break
                self._accumulate_side_batch(
                    self.model, self._prepare_inputs(inputs), side
                )
                trace.append((side, row))
        self.side_trace = trace
        return trace

    def _write_consumption_manifest(self, trace):
        path = Path(
            self.consumption_manifest_path
            or os.path.join(self.args.output_dir, "consumption_manifest.json")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "side_counts": {
                side: sum(1 for trace_side, _ in trace if trace_side == side)
                for side in ("forget", "retain")
            },
            "side_row_indices": {
                side: [row for trace_side, row in trace if trace_side == side]
                for side in ("forget", "retain")
            },
        }
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        return manifest

    def train(self, *args, **kwargs):
        if not self.independent_side_passes:
            return super().train(*args, **kwargs)

        del args, kwargs
        self.model.eval()
        self.model.zero_grad()
        trace = self._run_independent_side_passes()
        self._write_consumption_manifest(trace)
        self.state.global_step = 0
        self.save_model()
        return TrainOutput(global_step=0, training_loss=0.0, metrics={})

    # ------------------------------------------------------------------
    # Optimizer — no-op, we only accumulate gradient stats
    # ------------------------------------------------------------------

    def create_optimizer(self):
        # Use an actual model parameter so DeepSpeed ZeRO-3 can partition it.
        # lr=0.0 ensures no actual updates happen.
        base = self._unwrap(self.model)
        first_param = next(base.parameters())
        self.optimizer = torch.optim.SGD([first_param], lr=0.0)
        return self.optimizer

    # ------------------------------------------------------------------
    # Save continuous scores and optional Boolean support.
    # ------------------------------------------------------------------

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        num_select = int(self.select_frac * self.total_params)

        logger.info(f"Computing saliency scores with score_fn={self.score_fn!r}")

        # Compute saliency scores and gather on CPU for global threshold
        scores_flat = []
        for safe, _ in self._scored_params:
            score = self._compute_scores(safe)
            scores_flat.append(score.cpu().flatten())

        all_scores = torch.cat(scores_flat)

        # Log score distribution
        logger.info(
            f"Score stats: mean={all_scores.mean():.4f}, "
            f"std: {all_scores.std():.4f}, "
            f"min: {all_scores.min():.4f}, max: {all_scores.max():.4f}"
        )

        if not 0 < num_select <= len(all_scores):
            raise ValueError(
                "select_frac must select at least one and at most all params"
            )

        parameter_layout = [
            (original, self.forget_acc[safe].shape, self.forget_acc[safe].numel())
            for safe, original in self._scored_params
        ]
        if self.save_selector_logits:
            save_selector_logits(
                all_scores,
                parameter_layout,
                output_dir,
                selection_order="higher_is_selected",
                selector=self.score_fn.replace("_", ""),
                storage_dtype=getattr(self, "selector_logits_storage_dtype", "float16"),
            )

        # kthvalue is one-indexed. Select values strictly above the boundary,
        # then consume boundary ties in parameter/flattened-index order so the
        # global support is exact and deterministic.
        threshold = torch.kthvalue(
            all_scores, len(all_scores) - num_select + 1
        ).values.item()

        strictly_selected = 0
        for safe, _ in self._scored_params:
            strictly_selected += int(
                (self._compute_scores(safe) > threshold).sum().item()
            )
        ties_remaining = num_select - strictly_selected
        selected_masks = {}
        for safe, original in self._scored_params:
            score = self._compute_scores(safe)
            selected = score.cpu() > threshold
            equal = score.cpu() == threshold
            equal_count = int(equal.sum().item())
            if ties_remaining >= equal_count:
                selected |= equal
                ties_remaining -= equal_count
            elif ties_remaining > 0:
                indices = torch.nonzero(equal.flatten(), as_tuple=False).flatten()
                chosen = selected.flatten()
                chosen[indices[:ties_remaining]] = True
                ties_remaining = 0
            selected_masks[original] = selected
        num_selected = sum(int(value.sum().item()) for value in selected_masks.values())
        if num_selected != num_select or ties_remaining != 0:
            raise RuntimeError(
                f"Exact saliency budget failed: {num_selected} != {num_select}"
            )

        logger.info(
            f"Saliency mask: {num_selected:,}/{self.total_params:,} params "
            f"selected ({num_selected / self.total_params * 100:.2f}%, "
            f"target: {self.select_frac * 100:.1f}%)"
        )

        if self.save_boolean_active_mask:
            active_mask_path = os.path.join(output_dir, "active_mask.pt")
            torch.save(selected_masks, active_mask_path)
            logger.info(f"Saved boolean active mask to {active_mask_path}")

        # Optionally save raw accumulators (skip for large models to save disk)
        save_acc = os.environ.get("SAVE_ACCUMULATORS", "1") == "1"
        if save_acc:
            forget_acc_cpu = {k: v.cpu() for k, v in self.forget_acc.items()}
            retain_acc_cpu = {k: v.cpu() for k, v in self.retain_acc.items()}
            torch.save(forget_acc_cpu, os.path.join(output_dir, "forget_acc.pt"))
            torch.save(retain_acc_cpu, os.path.join(output_dir, "retain_acc.pt"))
            logger.info(f"Saved accumulators to {output_dir}")
        else:
            logger.info("Skipping accumulator save (SAVE_ACCUMULATORS=0)")
