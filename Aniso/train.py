from __future__ import annotations

import os
# 防止 PyTorch 多进程 mmap 共享内存失败
os.environ["TORCH_FORCE_SHARE_FD"] = "0"

import pickle
from typing import Any, Callable
from collections import Counter

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam
from Aniso.data import collate_fn

# 尝试引入 DeepSpeedCPUAdam（可选）
try:
    
    _HAS_DS_CPU_ADAM = True
except Exception:
    _HAS_DS_CPU_ADAM = False

# 保存路径设定（相对路径，可用环境变量 ANISO_SAVE_DIR 覆盖）
SAVE_ROOT = os.environ.get("ANISO_SAVE_DIR", "./aniso_logs")
SAVE_DIR = {
    "train": os.path.join(SAVE_ROOT, "train"),
    "val":   os.path.join(SAVE_ROOT, "val"),
    "test":  os.path.join(SAVE_ROOT, "test"),
}


class BaseLightning(pl.LightningModule):
    """Lightning wrapper defining dataset and training functions.

    折射率任务要点：
    - 模型输出仍为 3x3 张量（介电张量，中间层），不直接监督该张量；
    - 先做对称化 + 正定化（SPD 投影），取特征值；
    - 以 n = sqrt(mean(eigvals_pos)) 作为预测，损失只对 n 计算。
    """

    def __init__(
        self,
        dataset,
        model,
        num_workers: int = 0,
        batch_size: int = 32,
        loss_fn: Callable | None = None,
        additional_losses: dict | None = None,
        lr: float = 0.005,
        weight_decay: float = 0.0,
        use_deepspeed_cpu_adam: bool = False,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any | None = None,
        monitor: str | None = None,
        callbacks: list[pl.callbacks.callback.Callback] | None = None,
        # ===== 可调参数 =====
        log_beta: float = 0.05,               # SmoothL1 在 log(n) 空间的阈值
        aniso_reg_weight: float = 0.05,       # 各向异性正则权重 λ
        aniso_reg_cs: tuple[int, ...] = (6,), # 施加正则的晶系，默认仅 Cubic=6
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "dataset", "optimizer", "scheduler"])

        self.dataset = dataset
        self.model = model
        self.num_workers = num_workers
        self.batch_size = batch_size
                # —— 规范处理 DataLoader 的 worker 数 —— 
        # 传入 None → 自动给一个合适的默认值；传入 0/负数 → 单进程（最稳）
        # try:
        #     import os
        #     _AUTO_WORKERS = min(8, os.cpu_count() or 1)
        # except Exception:
        #     _AUTO_WORKERS = 1

        # if num_workers is None:
        #     self.num_workers = _AUTO_WORKERS
        # else:
        #     self.num_workers = int(num_workers)
        #     if self.num_workers < 0:
        #         self.num_workers = 0

        self.loss_fn = nn.functional.mse_loss if loss_fn is None else loss_fn
        self.additional_losses = additional_losses
        if additional_losses is None and self.loss_fn is nn.functional.mse_loss:
            self.additional_losses = {"mae": nn.functional.l1_loss}

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.monitor = monitor
        self.callbacks = callbacks

        if optimizer is not None:
            self.lr = optimizer.state_dict()["param_groups"][0].get("lr", lr)
            self.weight_decay = optimizer.state_dict()["param_groups"][0].get("weight_decay", weight_decay)
        else:
            self.lr = lr
            self.weight_decay = weight_decay

        self.use_deepspeed_cpu_adam = use_deepspeed_cpu_adam

        # 训练/正则参数
        self.log_beta = float(log_beta)
        self.aniso_reg_weight = float(aniso_reg_weight)
        self.aniso_reg_cs = tuple(int(x) for x in aniso_reg_cs)

        # 结果缓存
        self.saved_results = {"train": [], "val": [], "test": []}
        os.makedirs(SAVE_DIR["train"], exist_ok=True)
        os.makedirs(SAVE_DIR["val"],   exist_ok=True)
        os.makedirs(SAVE_DIR["test"],  exist_ok=True)

        # 初始化分晶系统计容器（避免 NoneType）
        self._val_abs_err_sum = None
        self._val_sample_cnt = None
        self._test_abs_err_sum = None
        self._test_sample_cnt = None

    # ---------------------------
    # SPD 投影 + n 聚合
    # ---------------------------
    @staticmethod
    def _spd_eigvals(cart_pred: torch.Tensor, eps: float = 1e-6):
        cart_sym = 0.5 * (cart_pred + cart_pred.transpose(-1, -2))   # [B,3,3]
        w, v = torch.linalg.eigh(cart_sym)                           # w:[B,3]
        w_pos = F.softplus(w) + eps                                  # 正值
        return w_pos, v

    @staticmethod
    def _n_from_eps(cart_pred: torch.Tensor):
        w_pos, _ = BaseLightning._spd_eigvals(cart_pred)             # [B,3]
        n_pred = torch.sqrt(w_pos.mean(dim=-1))                      # [B]
        return n_pred, w_pos

    def _get_losses(
        self,
        cart_pred: torch.Tensor,
        target_n: torch.Tensor,
        prefix: str,
        cs_idx: torch.LongTensor | None = None,
    ):
        """训练：加权 + 正则；验证/测试：不加权、不正则。"""
        n_pred, eigvals_pos = self._n_from_eps(cart_pred)
        y = target_n.to(n_pred.dtype).clamp(min=1e-6)

        base = F.smooth_l1_loss(torch.log(n_pred), torch.log(y), beta=self.log_beta, reduction="none")

        apply_weight = (prefix == "train") and (cs_idx is not None) and hasattr(self, "class_weights")
        if apply_weight:
            w_list = [self.class_weights.get(int(i.item()), 1.0) for i in cs_idx]
            weights = torch.tensor(w_list, dtype=base.dtype, device=base.device)
            loss_main = (base * weights).mean()
        else:
            loss_main = base.mean()

        # 各向异性一致性正则
        if (prefix == "train") and (cs_idx is not None) and (self.aniso_reg_weight > 0):
            mask = torch.zeros_like(cs_idx, dtype=torch.bool)
            for c in self.aniso_reg_cs:
                mask |= (cs_idx == int(c))
            if mask.any():
                raniso = eigvals_pos.var(dim=-1)
                loss_main = loss_main + self.aniso_reg_weight * raniso[mask].mean()

        losses = {
            f"{prefix}_loss": loss_main,
            f"{prefix}_mse":  F.mse_loss(n_pred, y),
            f"{prefix}_mae":  F.l1_loss(n_pred, y),
        }
        return losses, n_pred, eigvals_pos

    # ---------------------------
    # Lightning 标准接口
    # ---------------------------
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        cart_pred = self.model(batch)
        losses, n_pred, _ = self._get_losses(cart_pred, batch.target_n, prefix="train", cs_idx=batch.crystal_system_idx)
        self.log_dict(losses, on_epoch=True, on_step=False, batch_size=self.batch_size)
        return losses["train_loss"]

    def validation_step(self, batch, batch_idx):
        cart_pred = self.model(batch)
        losses, n_pred, _ = self._get_losses(cart_pred, batch.target_n, prefix="val", cs_idx=batch.crystal_system_idx)
        self.log_dict(losses, on_epoch=True, on_step=False, batch_size=self.batch_size)

        with torch.no_grad():
            y = batch.target_n.to(n_pred.dtype)
            abs_err = torch.abs(n_pred - y)
            for c in range(7):
                m = (batch.crystal_system_idx == c)
                if m.any():
                    self._val_abs_err_sum[c] += float(abs_err[m].sum().detach().cpu())
                    self._val_sample_cnt[c]  += int(m.sum().item())

    def test_step(self, batch, batch_idx):
        cart_pred = self.model(batch)
        losses, n_pred, _ = self._get_losses(cart_pred, batch.target_n, prefix="test", cs_idx=batch.crystal_system_idx)
        self.log_dict(losses, batch_size=self.batch_size)

        with torch.no_grad():
            y = batch.target_n.to(n_pred.dtype)
            abs_err = torch.abs(n_pred - y)
            for c in range(7):
                m = (batch.crystal_system_idx == c)
                if m.any():
                    self._test_abs_err_sum[c] += float(abs_err[m].sum().detach().cpu())
                    self._test_sample_cnt[c]  += int(m.sum().item())

    def configure_optimizers(self):
        if self.optimizer is not None:
            opt = self.optimizer
        else:
            use_deepspeed = False
            try:
                
                use_deepspeed = isinstance(getattr(self.trainer, "strategy", None), DeepSpeedStrategy)
            except Exception:
                pass

            if use_deepspeed and self.use_deepspeed_cpu_adam and _HAS_DS_CPU_ADAM:
                opt = DeepSpeedCPUAdam(
                    self.parameters(),
                    lr=self.lr, betas=(0.9, 0.999),
                    eps=1e-8, weight_decay=self.weight_decay
                )
            else:
                opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        sch = self.scheduler or optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
        
        if self.monitor is not None:
            return {
                "optimizer": opt,
                "lr_scheduler": sch,   # ← 关键：用 lr_scheduler
                "monitor": self.monitor,
            }


    def setup(self, stage: str = None):
        if isinstance(self.dataset, dict):
            self.train_dataset = self.dataset["train"]
            self.val_dataset   = self.dataset["valid"]
            self.test_dataset  = self.dataset["test"]
        else:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
            )

        cs_indices = [int(d.node_crystal_attr[0].argmax().item()) for d in self.train_dataset]
        counts = Counter(cs_indices)
        n_class = 7
        total = sum(counts.values()) if counts else 1

        raw = {cs: (total / (n_class * c)) for cs, c in counts.items() if c > 0}
        for cs in range(n_class):
            raw.setdefault(cs, 1.0)

        mean_w = float(np.mean(list(raw.values())))
        normed = {k: (v / mean_w) for k, v in raw.items()}
        W_MAX = 5.0
        self.class_weights = {k: min(v, W_MAX) for k, v in normed.items()}

        if getattr(self, "trainer", None) is None or getattr(self.trainer, "global_rank", 0) == 0:
            print("[class weights]", self.class_weights)
# 选择一个合理的默认（示例：min(16, os.cpu_count() or 8)）
    
    def train_dataloader(self) -> DataLoader:
        nw = self.num_workers
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=True,
            #persistent_workers=(nw > 0),
        )

    def val_dataloader(self) -> DataLoader:
        nw = self.num_workers
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=True,
            #persistent_workers=(nw > 0),
        )

    def test_dataloader(self) -> DataLoader:
        nw = self.num_workers
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=True,
            #persistent_workers=(nw > 0),
        )


    def configure_callbacks(self):
        if self.callbacks is None:
            lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
            early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=30, mode="min")
            ckpt = pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                filename="epoch{epoch:03d}-valloss{val_loss:.6f}",
                save_weights_only=False,
            )
            return [lr_cb, early_stop, ckpt]
        return self.callbacks

    # ---------------------------
    # 分晶系指标：按样本累计 + DDP 聚合
    # ---------------------------
    def on_validation_epoch_start(self):
        self._val_abs_err_sum = [0.0] * 7
        self._val_sample_cnt  = [0]   * 7

    def on_test_epoch_start(self):
        self._test_abs_err_sum = [0.0] * 7
        self._test_sample_cnt  = [0]   * 7

    def _gather_across_ranks(self, arr_floats, arr_ints):
        if getattr(self.trainer, "world_size", 1) == 1:
            return (torch.tensor(arr_floats, device=self.device, dtype=torch.float64),
                    torch.tensor(arr_ints,  device=self.device, dtype=torch.long))

        f = torch.tensor(arr_floats, device=self.device, dtype=torch.float64)
        i = torch.tensor(arr_ints,  device=self.device, dtype=torch.long)
        f_all = self.all_gather(f)  # [world, 7]
        i_all = self.all_gather(i)  # [world, 7]
        return f_all.sum(dim=0), i_all.sum(dim=0)

    def on_validation_epoch_end(self):
        sum_f, cnt_i = self._gather_across_ranks(self._val_abs_err_sum, self._val_sample_cnt)
        per_class_mae = (sum_f / cnt_i.clamp(min=1)).cpu().numpy().tolist()
        valid_mae = [mae for mae, cnt in zip(per_class_mae, cnt_i.tolist()) if cnt > 0]
        if len(valid_mae) > 0:
            macro_mae = float(np.mean(valid_mae))
            self.log("val_mae_macro", macro_mae, prog_bar=True, sync_dist=True)
        for c, (mae, cnt) in enumerate(zip(per_class_mae, cnt_i.tolist())):
            if cnt > 0:
                self.log(f"val_mae_cs{c}", float(mae), prog_bar=False, sync_dist=True)

    def on_test_epoch_end(self):
        sum_f, cnt_i = self._gather_across_ranks(self._test_abs_err_sum, self._test_sample_cnt)
        per_class_mae = (sum_f / cnt_i.clamp(min=1)).cpu().numpy().tolist()
        valid_mae = [mae for mae, cnt in zip(per_class_mae, cnt_i.tolist()) if cnt > 0]
        if len(valid_mae) > 0:
            macro_mae = float(np.mean(valid_mae))
            self.log("test_mae_macro", macro_mae, prog_bar=True, sync_dist=True)
        for c, (mae, cnt) in enumerate(zip(per_class_mae, cnt_i.tolist())):
            if cnt > 0:
                self.log(f"test_mae_cs{c}", float(mae), prog_bar=False, sync_dist=True)

