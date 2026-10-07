import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from training_access import validate_faithful_unlearning_config
from trainer.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    validate_faithful_unlearning_config(cfg)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    active_mask_path = cfg.get("active_mask_path", None)
    if active_mask_path:
        from trainer.mask_artifacts import apply_active_gradient_mask

        expected_version = cfg.get("active_mask_expected_version", None)
        if not expected_version:
            raise ValueError(
                "active_mask_expected_version is required with active_mask_path."
            )
        apply_active_gradient_mask(
            model,
            active_mask_path,
            expected_version=expected_version,
        )
    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Benchmark evaluators are deliberately not constructed until optimization
    # is over. This prevents eval_on_start/eval_strategy or a callback from
    # exposing official test metrics during model selection. A separately
    # declared validation dataset can still be used by the HF Trainer.
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    defer_benchmark_eval = bool(trainer_cfg.args.get("do_train", True))
    if eval_cfgs and not defer_benchmark_eval:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        if eval_cfgs and trainer.evaluators is None:
            trainer.evaluators = get_evaluators(
                eval_cfgs=eval_cfgs,
                template_args=template_args,
                model=model,
                tokenizer=tokenizer,
            )
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
