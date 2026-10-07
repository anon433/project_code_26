"""Continuous GD trajectory, observed without changing the native update."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import resource
import tempfile
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from trainer.unlearn.grad_diff import GradDiff
from trainer.mask_artifacts import save_active_mask_artifact, _checksum
from trainer.utils import compute_dpo_loss
from .scoring import select_global_topk, summarize_scores

VERSION = "tcus_trajectory_v1"
CHUNK = 1_000_000


def finite(tensor):
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), CHUNK):
        if not bool(torch.isfinite(flat[start : start + CHUNK]).all()):
            raise FloatingPointError(
                "Nonfinite TCUS trajectory tensor; no valid result"
            )


def batch_fingerprint(inputs):
    digest = hashlib.sha256()
    for side in ("forget", "retain"):
        for field in ("input_ids", "attention_mask", "labels"):
            values = (
                inputs[side][field]
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .contiguous()
            )
            digest.update(f"{side}/{field}:".encode())
            digest.update(values.numpy().tobytes())
    return digest.hexdigest()


class TrajectoryScores:
    """CPU FP32 gradients and scores; the positive part is per optimizer step."""

    def __init__(self, parameters, *, retain_penalty=1.0, probe_size=100000, seed=3):
        if not math.isfinite(retain_penalty) or retain_penalty < 0:
            raise ValueError("retain_penalty must be finite and nonnegative")
        self.retain_penalty = float(retain_penalty)
        self.scores = {
            n: torch.zeros(p.shape, dtype=torch.float32) for n, p in parameters.items()
        }
        self.forget = {n: torch.zeros_like(p) for n, p in self.scores.items()}
        self.retain = {n: torch.zeros_like(p) for n, p in self.scores.items()}
        self.snapshots = None
        self.completed_steps = 0
        self.microbatches = 0
        self.weight = 0.0
        self.history = []
        self.layout = []
        offset = 0
        for name, p in parameters.items():
            self.layout.append(
                dict(name=name, shape=list(p.shape), offset=offset, numel=p.numel())
            )
            offset += p.numel()
        indices = sorted(
            random.Random(seed).sample(range(offset), min(probe_size, offset))
        )
        self.probe_indices = torch.tensor(indices, dtype=torch.int64)
        self.probe_slices = {}
        from bisect import bisect_left

        for row in self.layout:
            lo = bisect_left(indices, row["offset"])
            hi = bisect_left(indices, row["offset"] + row["numel"])
            if hi > lo:
                self.probe_slices[row["name"]] = (
                    lo,
                    hi,
                    self.probe_indices[lo:hi] - row["offset"],
                )
        self.probe_forget = []
        self.probe_penalty = []

    @torch.no_grad()
    def add_microbatch(self, forget, retain, *, weight):
        if not math.isfinite(weight) or weight <= 0 or self.snapshots is not None:
            raise ValueError("Invalid trajectory microbatch weight or phase")
        for source, target in ((forget, self.forget), (retain, self.retain)):
            for name, grad in source.items():
                if grad is not None:
                    finite(grad)
                    target[name].add_(
                        grad.detach().to(device="cpu", dtype=torch.float32),
                        alpha=weight,
                    )
        self.microbatches += 1
        self.weight += weight

    @torch.no_grad()
    def before_step(self, parameters):
        if (
            self.snapshots is not None
            or not self.microbatches
            or not math.isclose(self.weight, 1.0)
        ):
            raise RuntimeError(
                "Trajectory requires one complete accumulation before update"
            )
        self.snapshots = {
            n: p.detach().to("cpu", copy=True) for n, p in parameters.items()
        }

    @torch.no_grad()
    def after_step(self, parameters):
        if self.snapshots is None:
            raise RuntimeError("Missing pre-update parameter snapshot")
        stats = dict(
            nonzero_displacement_count=0, forget_contribution=0.0, retain_penalty=0.0
        )
        pa = torch.zeros(self.probe_indices.numel())
        pc = torch.zeros_like(pa)
        for name, p in parameters.items():
            # Subtract after conversion: observe the representable parameter change.
            after = p.detach().to("cpu").reshape(-1)
            before = self.snapshots.pop(name).reshape(-1)
            gf, gr, score = (
                x[name].reshape(-1) for x in (self.forget, self.retain, self.scores)
            )
            probe = self.probe_slices.get(name)
            for start in range(0, p.numel(), CHUNK):
                stop = start + CHUNK
                delta = after[start:stop].float() - before[start:stop].float()
                a = gf[start:stop] * delta
                c = (gr[start:stop] * delta).clamp_min_(0)
                finite(a)
                finite(c)
                score[start:stop].add_(a).add_(c, alpha=-self.retain_penalty)
                finite(score[start:stop])
                stats["nonzero_displacement_count"] += int(torch.count_nonzero(delta))
                stats["forget_contribution"] += float(a.double().sum())
                stats["retain_penalty"] += float(c.double().sum())
                if probe is not None:
                    lo, hi, local = probe
                    mask = (local >= start) & (local < stop)
                    pa[lo:hi][mask] = a[local[mask] - start]
                    pc[lo:hi][mask] = c[local[mask] - start]
            self.forget[name].zero_()
            self.retain[name].zero_()
        self.snapshots = None
        self.completed_steps += 1
        stats.update(step=self.completed_steps, microbatches=self.microbatches)
        self.microbatches = 0
        self.weight = 0.0
        self.probe_forget.append(pa)
        self.probe_penalty.append(pc)
        self.history.append(stats)
        return stats


class _TrajectoryObserver(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        t = self.trainer
        if t.trajectory.microbatches != args.gradient_accumulation_steps:
            raise RuntimeError(
                "Partial gradient accumulation is unsupported in trajectory v1"
            )
        t.trajectory.before_step(t.scored_parameters)

    def on_optimizer_step(self, args, state, control, **kwargs):
        t = self.trainer
        if t.accelerator.optimizer_step_was_skipped:
            raise RuntimeError("Skipped trajectory optimizer step; no valid result")
        row = t.trajectory.after_step(t.scored_parameters)
        row["learning_rates"] = [g["lr"] for g in t.optimizer.param_groups]
        row["elapsed_seconds"] = time.monotonic() - t.started_at
        row["peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        row["gpu_peak_allocated_bytes"] = [
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        ]
        destination = Path(args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "trajectory_steps.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer.trajectory.completed_steps >= self.trainer.score_steps:
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        t = self.trainer
        if t.trajectory.completed_steps != t.score_steps or t.trajectory.microbatches:
            raise RuntimeError("TCUS trajectory incomplete; no valid result")
        t.search_complete = True
        t.trajectory.forget.clear()
        t.trajectory.retain.clear()


class TCUSTrajectoryGradDiff(GradDiff):
    """Native GD with read-only gradient diagnostics and optimizer-step observers."""

    def __init__(
        self,
        *args,
        score_steps=5,
        retain_penalty=1.0,
        budget_frac=0.1,
        param_filter="all",
        score_probe_size=100000,
        provenance_path=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if (
            not isinstance(score_steps, int)
            or isinstance(score_steps, bool)
            or not 0 < score_steps <= self.args.max_steps
        ):
            raise ValueError(
                "score_steps must be a positive integer within the full Stage 2 schedule"
            )
        if param_filter != "all" or self.retain_loss_type != "NLL":
            raise ValueError("Trajectory v1 requires all parameters and NLL retention")
        if (
            not 0 < budget_frac <= 1
            or not isinstance(score_probe_size, int)
            or score_probe_size <= 0
        ):
            raise ValueError("Invalid budget or probe size")
        if (
            self.args.world_size != 1
            or self.args.fp16
            or self.args.gradient_checkpointing
            or self.is_deepspeed_enabled
        ):
            raise ValueError(
                "Trajectory v1 requires one process, no fp16/DeepSpeed/checkpointing"
            )
        if self.args.save_strategy.value != "no" or self.args.do_eval:
            raise ValueError(
                "Trajectory v1 requires save_strategy=no and do_eval=false"
            )
        if getattr(self.model, "_active_gradient_mask_applied", False):
            raise ValueError("Trajectory search must start without a mask")
        self.scored_parameters = dict(self.model.named_parameters())
        if not all(p.requires_grad for p in self.scored_parameters.values()):
            raise ValueError("Trajectory search requires all parameters trainable")
        self.score_steps = score_steps
        self.budget_frac = float(budget_frac)
        self.provenance_path = provenance_path
        self.trajectory = TrajectoryScores(
            self.scored_parameters,
            retain_penalty=retain_penalty,
            probe_size=score_probe_size,
            seed=self.args.seed,
        )
        self.microbatch_count = 0
        self.consumption = []
        self.search_complete = False
        self.started_at = time.monotonic()
        self._training_started = False
        self.add_callback(_TrajectoryObserver(self))

    def train(self, *args, **kwargs):
        if self._training_started or args or kwargs.get("resume_from_checkpoint"):
            raise ValueError(
                "Trajectory v1 starts once from the original model; resume is unsupported"
            )
        self._training_started = True
        self.started_at = time.monotonic()
        return super().train(**kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if set(inputs) != {"forget", "retain"}:
            raise ValueError("Trajectory requires ordinary forget/retain batches")
        self.consumption.append(batch_fingerprint(inputs))
        forget_loss, retain_loss, outputs = self.compute_loss_components(model, inputs)
        params = tuple(self.scored_parameters.values())
        # Read diagnostics from the same forward graphs. autograd.grad does not
        # modify .grad; normal Trainer backward below remains the update source.
        gf = torch.autograd.grad(
            -forget_loss, params, retain_graph=True, allow_unused=True
        )
        gf = {
            n: g.detach().float().cpu()
            for n, g in zip(self.scored_parameters, gf)
            if g is not None
        }
        gr = torch.autograd.grad(
            retain_loss, params, retain_graph=True, allow_unused=True
        )
        gr = {
            n: g.detach().float().cpu()
            for n, g in zip(self.scored_parameters, gr)
            if g is not None
        }
        self.trajectory.add_microbatch(
            gf, gr, weight=1.0 / self.args.gradient_accumulation_steps
        )
        self.microbatch_count += 1
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        if (
            not self.search_complete
            or self.trajectory.completed_steps != self.score_steps
        ):
            raise RuntimeError("TCUS trajectory incomplete; refusing to publish mask")
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        selection = select_global_topk(
            self.trajectory.scores, budget_frac=self.budget_frac
        )
        provenance = (
            json.loads(Path(self.provenance_path).read_text())
            if self.provenance_path
            else {}
        )
        manifest = dict(
            algorithm="TCUS-Trajectory-GD",
            algorithm_version=VERSION,
            artifact_kind="active_mask",
            budget_frac=self.budget_frac,
            budget_scope="global",
            budget_mode="exact",
            param_filter="all",
            score_steps=self.score_steps,
            optimizer_steps=self.trajectory.completed_steps,
            microbatches=self.microbatch_count,
            retain_penalty=self.trajectory.retain_penalty,
            score_formula="sum(gf*delta - lambda*relu(gr*delta))",
            gradient_reduction="Stage2 microbatch accumulation before per-step positive part",
            objective_config=dict(
                alpha=self.alpha,
                gamma=self.gamma,
                retain_loss_type=self.retain_loss_type,
            ),
            selection_threshold=selection.threshold,
            positive_score_count=selection.positive_count,
            selected_count=selection.selected_count,
            total_count=selection.total_count,
            training_args=self.args.to_dict(),
            parameter_layout=self.trajectory.layout,
            provenance=provenance,
            scores_file="trajectory_scores.pt",
            model_mode="train",
        )
        with tempfile.TemporaryDirectory(
            prefix=".trajectory-", dir=output
        ) as temporary:
            temporary = Path(temporary)
            (temporary / "consumption.json").write_text(
                json.dumps(self.consumption) + "\n"
            )
            torch.save(self.trajectory.scores, temporary / "trajectory_scores.pt")
            torch.save(
                dict(
                    global_indices=self.trajectory.probe_indices,
                    parameter_layout=self.trajectory.layout,
                    forget=torch.stack(self.trajectory.probe_forget),
                    retain_penalty=torch.stack(self.trajectory.probe_penalty),
                ),
                temporary / "score_probe.pt",
            )
            summary = dict(
                score=summarize_scores(self.trajectory.scores),
                steps=self.trajectory.history,
            )
            (temporary / "score_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            manifest["scores_sha256"] = _checksum(temporary / "trajectory_scores.pt")
            save_active_mask_artifact(temporary, selection.active_mask, manifest)
            for path in sorted(
                temporary.iterdir(), key=lambda p: p.name == "mask_manifest.json"
            ):
                os.replace(path, output / path.name)


class TCUSTrajectoryNPO(TCUSTrajectoryGradDiff):
    """Observe native NPO updates with a frozen original-model reference."""

    def __init__(self, *args, beta=0.1, **kwargs):
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("NPO beta must be positive and finite")
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss_components(self, model, inputs):
        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=inputs["forget"],
            beta=self.beta,
        )
        retain_inputs = {
            key: inputs["retain"][key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        return forget_loss, retain_loss, forget_outputs

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        output = Path(output_dir or self.args.output_dir)
        manifest_path = output / "mask_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["algorithm"] = "TCUS-Trajectory-NPO"
        manifest["objective_config"].update(
            beta=self.beta, reference_model="frozen_initial_model"
        )
        manifest["forget_score_gradient"] = (
            "negative gradient of native NPO forget loss"
        )
        from src.reproduction.io import write_json

        write_json(manifest_path, manifest)
