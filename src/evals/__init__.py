from importlib import import_module

from omegaconf import DictConfig

EVALUATOR_REGISTRY = {
    "MUSEEvaluator": "evals.muse",
    "LMEvalEvaluator": "evals.lm_eval",
}


def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator_handler_name = eval_cfg.get("handler")
    assert evaluator_handler_name is not None, ValueError(f"{name} handler not set")
    module = EVALUATOR_REGISTRY.get(evaluator_handler_name)
    if module is None:
        raise NotImplementedError(
            f"{evaluator_handler_name} not implemented or not registered"
        )
    eval_handler = getattr(import_module(module), evaluator_handler_name)
    return eval_handler(eval_cfg, **kwargs)


def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators
