from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
import os
import torch


class SaveBestModelStateDict(Callback):
    def __init__(
        self,
        monitor: str,
        mode: str = "min",
        filename: str = "best_model_state_dict.pt",
    ):
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.best_score = float("inf") if mode == "min" else -float("inf")

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return  # Metric not available yet

        # Check if this is the best score so far
        if (self.mode == "min" and current < self.best_score) or (
            self.mode == "max" and current > self.best_score
        ):
            self.best_score = current

            # Determine artifact directory
            logger = trainer.logger
            if isinstance(logger, MLFlowLogger):
                model_dir = os.path.join(
                    logger.save_dir,
                    logger.experiment_id,
                    logger.run_id,
                    "artifacts",
                    "model",
                )
            else:
                model_dir = os.path.join(logger.save_dir, "model")

            os.makedirs(model_dir, exist_ok=True)

            # Save the model's state_dict
            model_path = os.path.join(model_dir, self.filename)
            torch.save(pl_module.state_dict(), model_path)


class InferenceTimer(Callback):
    """Times forward-only inference during test and logs it to MLflow.

    Logs (per scenario == per dispatch decision):
      - infer_time_total_s
      - infer_time_per_sample_s
    Toggles ReconstructionTask._time_forward so shared_step records CUDA-synced
    forward time + sample count (excludes loss/metric overhead).
    """

    def on_test_start(self, trainer, pl_module):
        pl_module._time_forward = True
        pl_module._infer_time_s = 0.0
        pl_module._infer_samples = 0

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        pl_module._time_forward = False
        n = int(getattr(pl_module, "_infer_samples", 0))
        if n <= 0:
            return
        total = float(pl_module._infer_time_s)
        metrics = {"infer_time_total_s": total, "infer_time_per_sample_s": total / n}
        logger = getattr(trainer, "logger", None)
        if isinstance(logger, MLFlowLogger):
            logger.log_metrics(metrics)
        print(f"[InferenceTimer] forward-only: {total:.4f}s / {n} samples "
              f"= {total / n * 1000:.4f} ms/sample")
