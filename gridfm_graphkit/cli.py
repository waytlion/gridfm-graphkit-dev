from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.hetero_powergrid_forecast_datamodule import LitGridHeteroForecastDataModule
from gridfm_graphkit.datasets.hetero_powergrid_temporal_datamodule import LitGridHeteroTemporalDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.training.callbacks import SaveBestModelStateDict
import numpy as np
import pandas as pd
import yaml
import torch
import random
import os
from gridfm_graphkit.io.param_handler import get_task

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only
import lightning as L


def get_training_callbacks(args):
    early_stop_callback = EarlyStopping(
        monitor="Validation loss",
        min_delta=args.callbacks.tol,
        patience=args.callbacks.patience,
        verbose=False,
        mode="min",
    )

    save_best_model_callback = SaveBestModelStateDict(
        monitor="Validation loss",
        mode="min",
        filename="best_model_state_dict.pt",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="Validation loss",  # or whichever metric you track
        mode="min",
        save_last=True,
        save_top_k=0,
    )
    timer=Timer()
    return [early_stop_callback, save_best_model_callback, checkpoint_callback, timer]


def get_best_model_state_dict_path(trainer):
    """Return path to the state_dict saved by SaveBestModelStateDict callback."""
    best_callback = None
    for callback in trainer.callbacks:
        if isinstance(callback, SaveBestModelStateDict):
            best_callback = callback
            break

    if best_callback is None:
        return None

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

    return os.path.join(model_dir, best_callback.filename)

@rank_zero_only
def log_tcn_receptive_field(logger, task):
    """
    - log recpetive fiel only when tcn encoder is used in ST-GNN
    - @rank_zero_only prevents duplicate logs in distributed runs
    """
    model = getattr(task, "model", task)
    tcn = getattr(model, "temporal_bus", None)
    if tcn is None:
        return
    logger.log_hyperparams({"tcn.receptive_field": int(tcn.receptive_field)})

def main_cli(args):
    if getattr(args, "tf32", False):
        torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere+ GPUs

    logger = MLFlowLogger(
        save_dir=args.log_dir,
        experiment_name=args.exp_name,
        run_name=args.run_name,
    )

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    config_args = NestedNamespace(**base_config)

    L.seed_everything(config_args.seed, workers=True)

    normalizer_stats_path = getattr(args, "normalizer_stats", None)
    if config_args.task.task_name in ["ForecastOPF"]:
        litGrid = LitGridHeteroForecastDataModule(
            config_args, args.data_path, normalizer_stats_path=normalizer_stats_path
        )
    elif config_args.task.task_name in ["ST_ForecastOPF"]:
        litGrid = LitGridHeteroTemporalDataModule(
            config_args, args.data_path, normalizer_stats_path=normalizer_stats_path
        )
    else:
        litGrid = LitGridHeteroDataModule(
            config_args, args.data_path, normalizer_stats_path=normalizer_stats_path
        )
    
    model = get_task(config_args, litGrid.data_normalizers)
    if args.command != "train":
        print(f"Loading model weights from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location="cpu")
        # Following is neccessary, if train was run with precompile .. -> Handle torch.compile checkpoints that save parameters under model._orig_mod.*
        if any(key.startswith("model._orig_mod.") for key in state_dict.keys()):
            state_dict = {
                key.replace("model._orig_mod.", "model.", 1): value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict)
    
    log_tcn_receptive_field(logger, model)

    precision = "bf16-true" if getattr(args, "bfloat16", False) else None
    if precision:
        print("Using bfloat16 precision (via Lightning Trainer precision='bf16-true')")

    compile_mode = getattr(args, "compile", None)
    if compile_mode is not None:
        if compile_mode in ("max-autotune", "max-autotune-no-cudagraphs"):
            # Allow ATen GEMM as fallback so Triton configs that exceed GPU
            # shared-memory limits (e.g. triton_mm OOM) are skipped gracefully
            # instead of causing autotuning errors.
            import torch._inductor.config as inductor_cfg

            inductor_cfg.max_autotune_gemm_backends = "ATEN,TRITON"
        print(f"Compiling model with torch.compile(mode='{compile_mode}')")
        model.model = torch.compile(model.model, mode=compile_mode, dynamic=False)

    trainer_kwargs = {}
    if precision:
        trainer_kwargs["precision"] = precision

    trainer = L.Trainer(
        logger=logger,
        accelerator=config_args.training.accelerator,
        devices=config_args.training.devices,
        strategy=config_args.training.strategy,
        log_every_n_steps=1000,
        default_root_dir=args.log_dir,
        max_epochs=config_args.training.epochs,
        callbacks=get_training_callbacks(config_args), 
        **trainer_kwargs,
    )
    if args.command == "train" or args.command == "finetune":
        trainer.fit(model=model, datamodule=litGrid)

        # Ensure evaluation/prediction uses the best validation model, not
        # the final in-memory weights from the last optimization step.
        trainer.strategy.barrier("load_best_model_state_dict")
        best_model_state_dict_path = get_best_model_state_dict_path(trainer)
        if best_model_state_dict_path is not None and os.path.exists(best_model_state_dict_path):
            print(f"Loading best model weights from {best_model_state_dict_path} for evaluation.")
            best_state_dict = torch.load(best_model_state_dict_path, map_location="cpu")
            model.load_state_dict(best_state_dict)
        else:
            print("Best model state_dict not found; using latest in-memory weights for evaluation.")

    if args.command != "predict":
        test_trainer = L.Trainer(
            logger=logger,
            accelerator=config_args.training.accelerator,
            devices=1,
            num_nodes=1,
            log_every_n_steps=1,
            default_root_dir=args.log_dir,
            **trainer_kwargs,
        )
        test_trainer.test(model=model, datamodule=litGrid)

    artifacts_dir = os.path.join(
        logger.save_dir, logger.experiment_id, logger.run_id, "artifacts"
    )

    compute_dc_ac = getattr(args, "compute_dc_ac_metrics", False)
    if compute_dc_ac:
        sn_mva = config_args.data.baseMVA
        from gridfm_graphkit.tasks.compute_ac_dc_metrics import compute_ac_dc_metrics
        for grid_name in config_args.data.networks:
            raw_dir = os.path.join(args.data_path, grid_name, "raw")
            print(f"\nComputing ground-truth AC/DC metrics for {grid_name}...")
            compute_ac_dc_metrics(artifacts_dir, raw_dir, grid_name, sn_mva)

    save_output = getattr(args, "save_output", False) or args.command == "predict"
    if save_output:
        if len(config_args.data.networks) > 1:
            raise NotImplementedError("Predict/save_output with multiple grids is not yet supported.")

        predict_trainer = L.Trainer(
            logger=logger,
            accelerator=config_args.training.accelerator,
            devices=1,
            num_nodes=1,
            log_every_n_steps=1,
            default_root_dir=args.log_dir,
            **trainer_kwargs,
        )
        predictions = predict_trainer.predict(model=model, datamodule=litGrid)

        rows = {key: [] for key in predictions[0].keys()}
        for batch in predictions:
            for key in rows:
                rows[key].append(batch[key])

        df = pd.DataFrame({key: np.concatenate(vals) for key, vals in rows.items()})

        grid_name = config_args.data.networks[0]
        if args.command == "predict":
            output_dir = args.output_path
        else:
            output_dir = os.path.join(artifacts_dir, "test")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{grid_name}_predictions.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved predictions to {out_path}")
