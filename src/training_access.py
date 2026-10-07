"""Research-integrity checks for optimization-time data access."""

from collections.abc import Mapping

from omegaconf import DictConfig, OmegaConf


_FORBIDDEN_METHOD_ARG_FRAGMENTS = (
    "retain_logs",
    "eval_logs",
    "reference_logs",
    "truth_ratio",
    "forget_quality",
    "fq_target",
    "holdout",
)
_FORBIDDEN_EVAL_ARTIFACT_NAMES = ("tofu_eval.json", "muse_eval.json")
_RETRAINED_MODEL_PATH_MARKERS = (
    "retain90",
    "retain95",
    "retain99",
    "_retrain",
    "-retrain",
)
_FORBIDDEN_ANSWER_KEYS = {
    "paraphrased_answer",
    "perturbed_answer",
}
_FORBIDDEN_QUESTION_KEYS = {"paraphrased_question"}


def _walk_config_values(value, path=()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_config_values(child, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_config_values(child, (*path, str(index)))
    else:
        yield path, value


def validate_faithful_unlearning_config(cfg: DictConfig):
    """Reject official benchmark signals from an unlearning optimizer."""
    if cfg.get("mode", "train") != "unlearn":
        return
    if not bool(cfg.trainer.args.get("do_train", True)):
        return

    method_args_cfg = cfg.trainer.get("method_args", {})
    method_args = (
        OmegaConf.to_container(method_args_cfg, resolve=True)
        if OmegaConf.is_config(method_args_cfg)
        else method_args_cfg
    )
    violations = []
    for path, value in _walk_config_values(method_args):
        dotted_method_path = ".".join(path)
        normalised_path = dotted_method_path.lower()
        if any(
            fragment in normalised_path for fragment in _FORBIDDEN_METHOD_ARG_FRAGMENTS
        ):
            violations.append(f"trainer.method_args.{'.'.join(path)}")
        if isinstance(value, str) and value.strip().lower().endswith(
            _FORBIDDEN_EVAL_ARTIFACT_NAMES
        ):
            violations.append(f"trainer.method_args.{'.'.join(path)}")

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    for path, value in _walk_config_values(data_cfg):
        if not path or not isinstance(value, str):
            continue
        normalised = value.strip().lower()
        dotted_path = f"data.{'.'.join(path)}"
        if normalised.endswith(_FORBIDDEN_EVAL_ARTIFACT_NAMES):
            violations.append(dotted_path)
        elif normalised in _FORBIDDEN_ANSWER_KEYS:
            violations.append(dotted_path)
        elif normalised in _FORBIDDEN_QUESTION_KEYS:
            violations.append(dotted_path)
        elif "holdout" in normalised or normalised.endswith("_perturbed"):
            violations.append(dotted_path)

    active_mask_path = cfg.get("active_mask_path")
    if active_mask_path and not cfg.get("active_mask_expected_version"):
        violations.append("active_mask_expected_version")
    if active_mask_path:
        weight_decay = float(cfg.trainer.args.get("weight_decay", 0.0))
        if weight_decay != 0.0:
            raise ValueError(
                "Faithful sparse-update violation: nonzero optimizer weight "
                "decay changes coordinates whose gradients are masked. Set "
                "trainer.args.weight_decay=0 for every masked run and its "
                "matched full-model control."
            )

    model_path = cfg.model.model_args.get("pretrained_model_name_or_path")
    if isinstance(model_path, str) and any(
        marker in model_path.lower() for marker in _RETRAINED_MODEL_PATH_MARKERS
    ):
        violations.append("model.model_args.pretrained_model_name_or_path")

    if violations:
        formatted = "\n  - ".join(sorted(set(violations)))
        raise ValueError(
            "Faithful unlearning boundary violation: official evaluation "
            "artifacts cannot be used during optimization. Move these inputs "
            f"to src/eval.py only:\n  - {formatted}"
        )
