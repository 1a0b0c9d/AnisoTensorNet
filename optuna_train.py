#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna 超参搜索（AnisoNet / matbench_dielectric，multi-fold 平均，monitor=val_loss）
使用 ddp_spawn，保证多卡各 rank 模型结构一致，避免 DDP 参数不匹配。
"""

import os
import gc
import json
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from joblib import dump as joblib_dump

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# 项目内模块
from Aniso.data import BaseDataset
from Aniso.model import E3nnModel
from Aniso.train import BaseLightning
from e3nn.io import CartesianTensor

# Matbench + pymatgen
from matbench.bench import MatbenchBenchmark
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ===== 全局设定 =====
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_FORCE_SHARE_FD", "0")
torch.set_default_dtype(torch.float64)

CRYSTAL_SYSTEMS = ["Triclinic","Monoclinic","Orthorhombic","Tetragonal","Trigonal","Hexagonal","Cubic"]

# ---------- 工具函数 ----------
def set_seeds(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def structure_to_ase_and_cs(pmg_struct: Structure, symprec: float = 0.01) -> Tuple[Any, str]:
    atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
    cs = SpacegroupAnalyzer(pmg_struct, symprec=symprec).get_crystal_system().capitalize()
    if cs.lower() == "rhombohedral":
        cs = "Trigonal"
    assert cs in CRYSTAL_SYSTEMS, f"Unexpected crystal system: {cs}"
    return atoms, cs

def make_df_from_inputs(inputs: List[Structure], outputs=None, subset_label="train", symprec: float = 0.01) -> pd.DataFrame:
    ase_list, cs_list, formulas = [], [], []
    for s in inputs:
        if not isinstance(s, Structure):
            raise TypeError("matbench_dielectric inputs must be pymatgen.Structure")
        a, cs = structure_to_ase_and_cs(s, symprec=symprec)
        ase_list.append(a)
        cs_list.append(cs)
        formulas.append(s.composition.reduced_formula)

    if outputs is None:
        n_vals = [0.0] * len(ase_list)
    else:
        n_vals = [float(x) for x in outputs]

    return pd.DataFrame({
        "formula": formulas,
        "structure": ase_list,
        "n": np.array(n_vals, dtype=np.float64),
        "crystal_system": cs_list,
        "subset": subset_label
    })

def stratified_split_by_crystal_system(df: pd.DataFrame, val_ratio=0.10, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cs, grp in df.groupby("crystal_system"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            train_idx.extend(idx)
            print(f"[stratified] crystal_system={cs}: only 1 sample → train.")
            continue
        n_val = max(1, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    df_tr = df.loc[train_idx].copy().reset_index(drop=True)
    df_va = df.loc[val_idx].copy().reset_index(drop=True)
    df_tr["subset"] = "train"; df_va["subset"] = "valid"
    return df_tr, df_va

# ----------------------------------------------------------------------------------------

def build_model_from_cfg(cfg: Dict[str, Any], ds_stats: BaseDataset) -> E3nnModel:
    ct = CartesianTensor("ij=ji")
    net = E3nnModel(
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
        max_radius=ds_stats.cutoff,            # 注意：这里使用 ds_stats
        number_of_basis=cfg["num_basis"],
        num_neighbors=ds_stats.num_neighbors,  # 以及这里
        num_nodes=getattr(ds_stats, "num_nodes", 1.0),
        reduce_output=True,
        same_em_layer=False,
    )
    return net

def one_fold_val_loss(
    fold_name: int,
    trial: optuna.trial.Trial,
    cfg: Dict[str, Any],
    trainer_args: Dict[str, Any],
    logdir: Path,
    symprec: float,
    stratified_val_ratio: float,
) -> float:
    mb = MatbenchBenchmark(subset=['matbench_dielectric'], autoload=False)
    task = list(mb.tasks)[0]
    task.load()
    X_train, y_train = task.get_train_and_val_data(fold_name)

    df_train_all = make_df_from_inputs(X_train, y_train, subset_label="train", symprec=symprec)
    df_tr, df_va = stratified_split_by_crystal_system(df_train_all, val_ratio=stratified_val_ratio, seed=42)

    ds_stats = BaseDataset(df_train_all, cutoff=cfg["cutoff"], symprec=symprec)
    ds_tr     = BaseDataset(df_tr,       cutoff=cfg["cutoff"], symprec=symprec)
    ds_va     = BaseDataset(df_va,       cutoff=cfg["cutoff"], symprec=symprec)

    net = build_model_from_cfg(cfg, ds_stats)
    lit = BaseLightning(
        dataset={"train": ds_tr, "valid": ds_va, "test": ds_va},
        model=net,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        optimizer=None,
        scheduler=None,
        monitor="val_loss",
        callbacks=None,
        log_beta=cfg["log_beta"],
        aniso_reg_weight=cfg["aniso_reg_weight"],
        aniso_reg_cs=(6,),
    )

    run_name = f"trial_{trial.number}_fold_{fold_name}"
    trial_dir = Path(logdir) / run_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    # 只在 rank0 记录日志与保存 ckpt（ddp_spawn 下 Lightning 会自动只在主进程写）
    logger = CSVLogger(save_dir=str(logdir), name=run_name)
    ckpt = ModelCheckpoint(
        dirpath=str(trial_dir),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        filename="epoch{epoch:03d}-valloss{val_loss:.6f}",
        save_weights_only=False,
    )
    early = EarlyStopping(monitor="val_loss", patience=30, mode="min")
    lrmon = LearningRateMonitor(logging_interval="epoch")
    prune_cb = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    trainer = Trainer(
        max_epochs=trainer_args["max_epochs"],
        accelerator=trainer_args["accelerator"],
        devices=trainer_args["devices"],
        strategy=trainer_args["strategy"],   # ← ddp_spawn
        enable_progress_bar=trainer_args["enable_progress_bar"],
        logger=logger,
        callbacks=[ckpt, early, lrmon, prune_cb],
        num_sanity_val_steps=0,
        detect_anomaly=False,
        deterministic=True,
        enable_model_summary=False,
    )

    trainer.fit(lit)
    val_loss = float(trainer.callback_metrics["val_loss"].item())

    try:
        trainer.strategy.teardown()
    except Exception:
        pass
    del trainer, lit, net, ds_tr, ds_va, ds_stats
    torch.cuda.empty_cache()
    gc.collect()
    return val_loss

def build_search_space(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "em_dim": trial.suggest_categorical("em_dim", [8, 16, 32, 48]),
        "em_attr_dim": trial.suggest_categorical("em_attr_dim", [16, 32, 48, 64]),
        "em_crystal_dim": trial.suggest_categorical("em_crystal_dim", [16, 32, 48, 64]),
        "layers": trial.suggest_int("layers", 1, 3),
        "mul": trial.suggest_int("mul", 16, 64, step=16),
        "lmax": trial.suggest_int("lmax", 2, 4),
        "num_basis": trial.suggest_int("num_basis", 10, 25, step=5),
        "batch_size": trial.suggest_categorical("batch_size", [8, 12, 16, 24, 32]),
        "cutoff": trial.suggest_float("cutoff", 4.0, 6.0, step=0.5),
        "log_beta": trial.suggest_float("log_beta", 0.02, 0.20),
        "aniso_reg_weight": trial.suggest_float("aniso_reg_weight", 0.0, 0.15),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", type=str, default="aniso_dielectric_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--enable-progress-bar", action="store_true", default=False)
    parser.add_argument("--folds", nargs="*", default=["0"])
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--logdir", type=str, default="./optuna_logs")
    parser.add_argument("--save-prefix", type=str, default="./optuna_outputs")
    args = parser.parse_args()

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    Path(args.save_prefix).mkdir(parents=True, exist_ok=True)

    # 全局随机性；包括 dataloader worker
    set_seeds(args.seed)
    pl.seed_everything(args.seed, workers=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=args.storage,
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # 关键：使用 ddp_spawn，保证各 rank 拿到的是同一份已构建好的 LightningModule
    strategy = "ddp_spawn" if (accelerator == "gpu" and args.devices > 1) else "auto"

    trainer_args = dict(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=(args.devices if accelerator == "gpu" else 1),
        strategy=strategy,
        enable_progress_bar=args.enable_progress_bar,
    )

    folds = [int(f) for f in args.folds]

    def objective(trial: optuna.trial.Trial) -> float:
        cfg = build_search_space(trial)
        val_losses = []
        for f in folds:
            loss_f = one_fold_val_loss(
                f, trial, cfg, trainer_args,
                Path(args.logdir), args.symprec, args.val_ratio
            )
            val_losses.append(loss_f)
            trial.report(float(np.mean(val_losses)), step=len(val_losses))
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(val_losses))

    study.optimize(objective, n_trials=args.n_trials, n_jobs=1, show_progress_bar=True)

    best = {
        "best_value_mean_val_loss": study.best_value,
        "best_params": study.best_params,
        "best_trial": int(study.best_trial.number),
        "folds": folds,
        "max_epochs": args.max_epochs,
    }
    best_json = Path(args.save_prefix) / f"{args.study_name}_best.json"
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    df = study.trials_dataframe()
    df.to_csv(Path(args.save_prefix) / f"{args.study_name}_trials.csv", index=False)
    joblib_dump(study, Path(args.save_prefix) / f"{args.study_name}.pkl")

    print("\n===== Optuna Done =====")
    print("Best value (mean val_loss across folds):", best["best_value_mean_val_loss"])
    print("Best params:", best["best_params"])
    print(f"Saved best to: {best_json}")

if __name__ == "__main__":
    main()
