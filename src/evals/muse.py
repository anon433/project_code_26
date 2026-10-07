from evals.base import Evaluator


class MUSEEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("MUSE", eval_cfg, **kwargs)

    def load_logs_from_file(self, file):
        logs = super().load_logs_from_file(file)
        unexpected = sorted(set(logs) - set(self.metrics))
        if unexpected:
            raise ValueError(
                "MUSE evaluation artifact contains non-requested metrics: "
                f"{unexpected}."
            )
        return logs

    def save_logs(self, logs, file):
        requested_logs = {
            metric_name: logs[metric_name]
            for metric_name in self.metrics
            if metric_name in logs
        }
        super().save_logs(requested_logs, file)
