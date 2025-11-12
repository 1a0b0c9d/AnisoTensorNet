#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna Mini-Test for AnisoNet (matbench_dielectric)

只跑 1 个 fold、2 个 trial、每个 trial 最多 2 个 epoch。
用于快速验证 pipeline 是否可运行。
"""

import os
import gc
import json
import random
from pathlib import Path
import numpy as np
import torch
import optuna
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# 项目模块
from Aniso.data import BaseDataset
from Aniso.model import E3nnModel
from Aniso.train import BaseLightning
from run_matbench_AnisoTensorNet import make_df_from_inputs, stratified_split_by_crystal_system, set_seeds
from e3nn.io import CartesianTensor
from matbench.bench import MatbenchBenchmark


# ========= Config =========
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_FORCE_SHARE_FD", "0")
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("./optuna_minitest_logs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def build_model(cfg, ds_stats):
    """构建模型"""
    ct = CartesianTensor("ij=ji")
    return E3nnModel(
        in_dim=118,
        in_attr_dim=118,
        em_dim=cfg["em_dim"],
        em_attr_dim=cfg["em_attr_dim"],
        crystal_dim=7,
        em_crystal_dim=cfg["em_crystal_dim"],
        irreps_out=str(ct),
        layers=cfg["layers"],
        mul=cfg["mul"],
        lmax=cfg["lmax"],
        max_radius=ds_stats.cutoff,
        number_of_basis=cfg["num_basis"],
        num_neighbors=ds_stats.num_neighbors,
        num_nodes=getattr(ds_stats, "num_nodes", 1.0),
        reduce_output=True,
        same_em_layer=False,
    )


def objective(trial):
    set_seeds(42)

    # 超参数空间（缩小范围）
    cfg = {
        "lr": trial.suggest_float("lr", 1e-3, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
        "em_dim": trial.suggest_categorical("em_dim", [8, 16]),
        "em_attr_dim": trial.suggest_categorical("em_attr_dim", [16, 32]),
        "em_crystal_dim": trial.suggest_categorical("em_crystal_dim", [16]),
        "layers": 1,
        "mul": 16,
        "lmax": 2,
        "num_basis": 10,
        "batch_size": 8,
        "cutoff": 5.0,
    }

    # 数据
    mb = MatbenchBenchmark(subset=["matbench_dielectric"], autoload=False)
    task = list(mb.tasks)[0]   # 取第一个任务

    task.load()
    fold =0
    X_train, y_train = task.get_train_and_val_data(fold)
    df_all = make_df_from_inputs(X_train, y_train, "train")
    df_tr, df_va = stratified_split_by_crystal_system(df_all, val_ratio=0.1, seed=42)

    ds_stats = BaseDataset(df_all, cutoff=cfg["cutoff"], symprec=0.01)
    ds_tr = BaseDataset(df_tr, cutoff=cfg["cutoff"], symprec=0.01)
    ds_va = BaseDataset(df_va, cutoff=cfg["cutoff"], symprec=0.01)

    # 模型
    net = build_model(cfg, ds_stats)
    lit = BaseLightning(
        dataset={"train": ds_tr, "valid": ds_va, "test": ds_va},
        model=net,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        monitor="val_loss",
    )

    # 回调 & logger
    trial_dir = OUTPUT_DIR / f"trial_{trial.number}"
    trial_dir.mkdir(exist_ok=True)
    ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=trial_dir)
    early = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    logger = CSVLogger(save_dir=str(OUTPUT_DIR), name=f"trial_{trial.number}")

    trainer = Trainer(
        max_epochs=2,   # mini-test 只跑 2 epoch
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[ckpt, early],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )
    trainer.fit(lit)

    val_loss = trainer.callback_metrics["val_loss"].item()

    # 清理资源
    try:
        trainer.strategy.teardown()
    except Exception:
        pass
    del trainer, lit, net, ds_tr, ds_va, ds_stats
    torch.cuda.empty_cache()
    gc.collect()

    return val_loss


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)  # mini-test: 2 个 trial

    print("\n===== Mini Test Done =====")
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
