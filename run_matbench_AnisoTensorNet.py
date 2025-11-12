import os
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
import lightning.pytorch as pl

# ① 提前设置默认精度（在导入 Aniso.* 前）
torch.set_default_dtype(torch.float64)

# 相对导入你在本目录下打包的最小模块
from Aniso.data import BaseDataset, collate_fn
from Aniso.model import E3nnModel
from Aniso.train import BaseLightning
from torch.utils.data import DataLoader
from e3nn.io import CartesianTensor
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from matbench.bench import MatbenchBenchmark
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd


CRYSTAL_SYSTEMS = ["Triclinic","Monoclinic","Orthorhombic","Tetragonal","Trigonal","Hexagonal","Cubic"]


def structure_to_ase_and_cs(pmg_struct, symprec=0.01):
    """pymatgen.Structure -> (ASE Atoms, crystal_system str)"""
    atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
    cs = SpacegroupAnalyzer(pmg_struct, symprec=symprec).get_crystal_system().capitalize()
    if cs.lower() == "rhombohedral":
        cs = "Trigonal"
    assert cs in CRYSTAL_SYSTEMS, f"Unexpected crystal system: {cs}"
    return atoms, cs


def make_df_from_inputs(inputs, outputs=None, subset_label="train"):
    """把 Matbench 的 inputs(list[Structure]) / outputs(np.ndarray|list|None) 转成 BaseDataset 需要的 DataFrame"""
    ase_list, cs_list = [], []
    for s in tqdm(inputs, desc=f"prep[{subset_label}]"):
        if not isinstance(s, Structure):
            raise TypeError("matbench_dielectric inputs must be pymatgen.Structure")
        a, cs = structure_to_ase_and_cs(s)
        ase_list.append(a)
        cs_list.append(cs)

    if outputs is None:
        n_vals = [0.0] * len(ase_list)  # 预测阶段占位
    else:
        n_vals = [float(x) for x in outputs]

    df = pd.DataFrame({
        "formula": [s.composition.reduced_formula for s in inputs],
        "structure": ase_list,
        "n": np.array(n_vals, dtype=np.float64),
        "crystal_system": cs_list,
        "subset": subset_label
    })
    return df


def stratified_split_by_crystal_system(df: pd.DataFrame, val_ratio=0.05, seed: int = 42):
    """按 crystal_system 分层切分，带小样本兜底"""
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cs, grp in df.groupby("crystal_system"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            # 无法同时覆盖 train/val，优先保证 train 稳定
            train_idx.extend(idx)
            print(f"[stratified] crystal_system={cs}: only 1 sample → go to train.")
            continue
        # 正常分层：至少 1 个进 val，且不能把该晶系全分到 val
        n_val = max(1, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    df_train_split = df.loc[train_idx].copy().reset_index(drop=True)
    df_val_split   = df.loc[val_idx].copy().reset_index(drop=True)
    df_train_split["subset"] = "train"
    df_val_split["subset"]   = "valid"
    return df_train_split, df_val_split


class AnisoFoldRunner:
    """每个 fold 训练一次并用于预测"""
    def __init__(self, cfg, fold_name: str):
        self.cfg = cfg
        self.fold_name = fold_name
        self.model = None
        self.lit = None
        self.best_ckpt_path = None

    def _build_model(self, ds_train_stats):
        ct = CartesianTensor("ij=ji")
        net = E3nnModel(
            in_dim=118,
            in_attr_dim=118,
            em_dim=self.cfg.em_dim,
            em_attr_dim=self.cfg.em_attr_dim,
            crystal_dim=7,
            em_crystal_dim=self.cfg.em_crystal_dim,
            irreps_out=str(ct),
            layers=self.cfg.layers,
            mul=self.cfg.mul,
            lmax=self.cfg.lmax,
            max_radius=ds_train_stats.cutoff,
            number_of_basis=self.cfg.num_basis,
            num_neighbors=ds_train_stats.num_neighbors,
            num_nodes=getattr(ds_train_stats, "num_nodes", 1.0),
            reduce_output=True,
            same_em_layer=False,
        )
        return net

    def train(self, X_train, y_train):
        # 1) 组装 DataFrame
        df_train = make_df_from_inputs(X_train, y_train, subset_label="train")

        # 2) 分层切分 train/val
        df_tr, df_val = stratified_split_by_crystal_system(df_train, val_ratio=0.10, seed=42)
        ds_train_full = BaseDataset(df_train, cutoff=self.cfg.cutoff, symprec=self.cfg.symprec)  # 统计用
        ds_train = BaseDataset(df_tr,  cutoff=self.cfg.cutoff, symprec=self.cfg.symprec)
        ds_val   = BaseDataset(df_val, cutoff=self.cfg.cutoff, symprec=self.cfg.symprec)

        # 3) Model & Lightning
        net = self._build_model(ds_train_full)

        # 每 fold 独立目录（防覆盖）
        ckpt_dir = os.path.join("checkpoints", f"{self.fold_name}")
        os.makedirs(ckpt_dir, exist_ok=True)
        log_name = f"aniso_{self.fold_name}"

        # 交由 BaseLightning 使用我们传入的 callbacks（避免重复）
        mc = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch{epoch:03d}-valloss{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            save_weights_only=False,  # 保存 full ckpt，便于完整恢复
        )
        early = pl.callbacks.EarlyStopping(monitor="val_loss", patience=45, mode="min")
        lrmon = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

        lit = BaseLightning(
            dataset={"train": ds_train, "valid": ds_val, "test": ds_val},
            model=net,
            batch_size=self.cfg.batch_size,
            lr=self.cfg.lr,
            weight_decay=self.cfg.wd,
            optimizer=None,                    # 由 BaseLightning 内部创建（也可外部自建传入）
            scheduler=None,
            monitor="val_loss",
            callbacks=[lrmon, early, mc],      # ← 只用这一套，避免 Trainer 再传一遍
        )

        # 4) 训练
        trainer = Trainer(
            max_epochs=self.cfg.epochs,
            accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
            enable_progress_bar=True,
            logger=CSVLogger("logs", name=log_name),
            num_sanity_val_steps=0,
            devices=1,
        )
        trainer.fit(lit)

        # 记录 best ckpt 路径
        self.best_ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
        if not self.best_ckpt_path or not os.path.isfile(self.best_ckpt_path):
            # 回退到最后一次
            if hasattr(trainer.checkpoint_callback, "last_model_path") and \
               os.path.isfile(trainer.checkpoint_callback.last_model_path):
                self.best_ckpt_path = trainer.checkpoint_callback.last_model_path
        print(f"✓ [{self.fold_name}] best checkpoint: {self.best_ckpt_path}")

        # 用官方推荐方式完整恢复（需要 full ckpt）
        self.lit = BaseLightning.load_from_checkpoint(
            checkpoint_path=self.best_ckpt_path,
            dataset={"train": ds_train, "valid": ds_val, "test": ds_val},
            model=net,                     # 提供同构的实例，方便跨设备加载
            batch_size=self.cfg.batch_size,
            lr=self.cfg.lr,
            weight_decay=self.cfg.wd,
            optimizer=None,
            scheduler=None,
            monitor="val_loss",
            callbacks=None,                # 仅推理阶段不再需要回调
        )
        self.model = self.lit.model.eval()
        return self

    @torch.no_grad()
    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call train() first.")
        df_test = make_df_from_inputs(X_test, outputs=None, subset_label="test")
        ds_test = BaseDataset(df_test, cutoff=self.cfg.cutoff, symprec=self.cfg.symprec)

        loader = DataLoader(ds_test, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=collate_fn)

        preds = []
        device = next(self.model.parameters()).device
        for batch in loader:
            batch.to(device)
            cart_pred = self.model(batch)               # [B,3,3], double
            n_pred, _ = BaseLightning._n_from_eps(cart_pred)
            preds.extend(n_pred.detach().cpu().numpy().astype("float64").tolist())
        return np.array(preds, dtype=np.float64)

    def get_params(self):
        # 只返回“原生类型”的小字典，便于记录
        return {
            "em_dim": self.cfg.em_dim,
            "em_attr_dim": self.cfg.em_attr_dim,
            "em_crystal_dim": self.cfg.em_crystal_dim,
            "layers": self.cfg.layers,
            "lmax": self.cfg.lmax,
            "num_basis": self.cfg.num_basis,
            "mul": self.cfg.mul,
            "lr": self.cfg.lr,
            "wd": self.cfg.wd,
            "batch_size": self.cfg.batch_size,
            "epochs": self.cfg.epochs,
            "cutoff": self.cfg.cutoff,
            "symprec": self.cfg.symprec,
        }


def set_seeds(seed: int = 1234):
    """④ 固定随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001794623322242499)
    parser.add_argument("--wd", type=float, default=0.05001627627802547)
    parser.add_argument("--em_dim", type=int, default=8)
    parser.add_argument("--em_attr_dim", type=int, default=32)
    parser.add_argument("--em_crystal_dim", type=int, default=48)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--num_basis", type=int, default=15)
    parser.add_argument("--mul", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--symprec", type=float, default=0.01)
    args = parser.parse_args()

    # ④ 设定随机种子（尽早调用）
    set_seeds(1234)

    class Cfg: pass
    cfg = Cfg();  [setattr(cfg, k, getattr(args, k)) for k in vars(args)]

    # 只跑 dielectric 子任务
    mb = MatbenchBenchmark(subset=['matbench_dielectric'], autoload=False)

    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            print(f"\n=== Fold {fold} ===")
            X_train, y_train = task.get_train_and_val_data(fold)
            X_test = task.get_test_data(fold, include_target=False)

            # fold 名字直接用 Matbench 的 fold 字符串，保证与记录一致
            runner = AnisoFoldRunner(cfg, fold_name=str(fold)).train(X_train, y_train)
            y_pred = runner.predict(X_test)

            # 长度&类型防御
            assert len(y_pred) == len(X_test)
            y_pred = np.asarray(y_pred, dtype=np.float64)

            task.record(fold, y_pred, params=runner.get_params())

    # 导出 Matbench 结果文件（提交要用）
    out_path = "results.json.gz"
    mb.to_file(out_path)
    print(f"\n✅ wrote {out_path}")


if __name__ == "__main__":
    main()
