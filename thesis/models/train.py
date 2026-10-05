"""Training loop for thesis CNN distinguisher with validation AUC early stopping."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Optional

import numpy as np

from ciphers.registry import max_rounds
from thesis.eval.metrics import classification_metrics, select_validation_threshold
from thesis.models.cnn_distinguisher import CnnDistinguisher, build_model


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested in the training configuration, but PyTorch "
            "does not report an available CUDA device"
        )
    return device


def _validate_train_config(cfg: "TrainConfig") -> None:
    if cfg.epochs < 1:
        raise ValueError("epochs must be positive")
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if cfg.patience < 1:
        raise ValueError("patience must be positive")
    if not np.isfinite(cfg.lr) or cfg.lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if not np.isfinite(cfg.weight_decay) or cfg.weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    if cfg.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if len(cfg.channels) not in (2, 3) or any(
        isinstance(channel, bool) or not isinstance(channel, int) or channel < 1
        for channel in cfg.channels
    ):
        raise ValueError("channels must contain two or three positive integers")


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    device: str = "auto"
    seed: int = 1
    channels: tuple[int, ...] = (32, 64, 128)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        if "channels" in filtered and isinstance(filtered["channels"], list):
            filtered["channels"] = tuple(filtered["channels"])
        return cls(**filtered)


def _index_subset(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X[idx], y[idx]


def _validate_training_data(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
) -> None:
    if X.ndim != 2 or X.shape[1] != 64 or not np.isfinite(X).all():
        raise ValueError("X must be a finite (N, 64) feature matrix")
    if not np.isin(X, (0.0, 1.0)).all():
        raise ValueError("X must contain binary ciphertext features")
    if y.ndim != 1 or len(y) != len(X) or not np.isin(y, (0, 1)).all():
        raise ValueError("y must be a binary vector aligned with X")
    expected_names = {"train", "val", "test"}
    if set(splits) != expected_names:
        raise ValueError(f"splits must contain exactly {sorted(expected_names)}")

    seen: set[int] = set()
    for name in ("train", "val", "test"):
        indices = np.asarray(splits[name])
        if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
            raise ValueError(f"{name} split must be a non-empty integer vector")
        values = [int(index) for index in indices]
        if min(values) < 0 or max(values) >= len(y):
            raise ValueError(f"{name} split contains out-of-range indices")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} split contains duplicate indices")
        overlap = seen.intersection(values)
        if overlap:
            raise ValueError(f"data leakage: {name} split overlaps an earlier split")
        if set(np.unique(y[indices])) != {0, 1}:
            raise ValueError(f"{name} split must contain both classes")
        seen.update(values)
    if len(seen) != len(y):
        raise ValueError("splits must partition every sample exactly once")


def _batch_iter(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n = len(y)
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        sl = order[start : start + batch_size]
        yield X[sl], y[sl]


def _forward_loss(
    model: CnnDistinguisher,
    xb: np.ndarray,
    yb: np.ndarray,
    device: str,
    loss_fn: Any,
) -> Any:
    import torch

    xt = torch.from_numpy(xb).to(device)
    yt = torch.from_numpy(yb.astype(np.float32)).to(device)
    logits = model.forward_logits(xt).squeeze(-1)
    return loss_fn(logits, yt)


def _predict_scores(model: CnnDistinguisher, X: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        scores: list[np.ndarray] = []
        bs = 4096
        for start in range(0, len(X), bs):
            xb = X[start : start + bs]
            xt = torch.from_numpy(xb.astype(np.float32)).to(device)
            prob = model.predict_proba(xt)
            scores.append(prob.cpu().numpy())
        return np.concatenate(scores, axis=0)


def train_distinguisher(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    cipher: str,
    rounds: int,
    model_dir: Path | str,
    cfg: Optional[TrainConfig] = None,
    log_dir: Optional[Path | str] = None,
    checkpoint_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Train CNN with early stopping on validation AUC; save best checkpoint.
    
    Parameters
    ----------
    log_dir : Path | str, optional
        If provided, log metrics to TensorBoard at this directory.
        Layout: log_dir/seed_{seed}/{cipher}_R{rounds}/
    """
    import torch
    import torch.nn as nn

    train_cfg = cfg or TrainConfig()
    _validate_train_config(train_cfg)
    if cipher not in {"simon", "speck"}:
        raise ValueError(f"unsupported cipher {cipher!r}")
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise ValueError("rounds must be an integer")
    if not 1 <= rounds <= max_rounds(cipher):
        raise ValueError(
            f"rounds must be in 1..{max_rounds(cipher)} for {cipher}"
        )
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    _validate_training_data(X, y, splits)
    device = _resolve_device(train_cfg.device)
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_dir = Path(model_dir)
    run_id = str((checkpoint_metadata or {}).get("run_id", "unscoped"))
    seed_model_dir = model_dir / run_id / f"seed_{train_cfg.seed}" / cipher
    seed_model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = seed_model_dir / f"R{rounds}.pt"

    writer = None
    tb_log_dir: Path | None = None
    X_tr, y_tr = _index_subset(X, y, splits["train"])
    X_va, y_va = _index_subset(X, y, splits["val"])

    model = build_model(train_cfg.channels).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(train_cfg.seed)
    best_auc = -1.0
    patience_ctr = 0
    best_state: Optional[dict[str, Any]] = None
    history: list[dict[str, float]] = []

    if log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging was requested but tensorboard is not "
                "installed. Run: python -m pip install -r requirements-thesis.txt"
            ) from exc
        log_dir = Path(log_dir)
        tb_log_dir = (
            log_dir / run_id / f"seed_{train_cfg.seed}" / f"{cipher}_R{rounds}"
        )
        writer = SummaryWriter(log_dir=str(tb_log_dir))

    try:
        for epoch in range(train_cfg.epochs):
            model.train()
            epoch_losses: list[float] = []
            for xb, yb in _batch_iter(X_tr, y_tr, train_cfg.batch_size, rng):
                opt.zero_grad()
                loss = _forward_loss(model, xb, yb, device, loss_fn)
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.item()))

            val_scores = _predict_scores(model, X_va, device)
            val_m = classification_metrics(y_va, val_scores)
            train_loss_mean = float(np.mean(epoch_losses))
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss_mean,
                    "val_auc": val_m["auc_roc"],
                    "val_accuracy": val_m["accuracy"],
                    "val_advantage_edge": val_m["advantage_edge"],
                }
            )

            if writer is not None:
                writer.add_scalar("Loss/Train", train_loss_mean, epoch)
                writer.add_scalar("AUC/Validation", val_m["auc_roc"], epoch)
                writer.add_scalar("Accuracy/Validation", val_m["accuracy"], epoch)
                writer.add_scalar(
                    "AdvantageEdge/Validation",
                    val_m["advantage_edge"],
                    epoch,
                )

            if val_m["auc_roc"] > best_auc:
                best_auc = val_m["auc_roc"]
                patience_ctr = 0
                best_state = {
                    key: value.cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                patience_ctr += 1
                if patience_ctr >= train_cfg.patience and best_state is not None:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        validation_scores = _predict_scores(model, X_va, device)
        decision_threshold = select_validation_threshold(y_va, validation_scores)
    finally:
        if writer is not None:
            writer.close()

    checkpoint = {
        "schema_version": 2,
        "state_dict": model.state_dict(),
        "cipher": cipher,
        "rounds": rounds,
        "train_config": train_cfg.__dict__,
        "best_val_auc": best_auc,
        "history": history,
        "decision_threshold": decision_threshold,
        "experiment": checkpoint_metadata or {},
    }
    temporary = ckpt_path.with_name(f".{ckpt_path.name}.tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(ckpt_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "model": model,
        "checkpoint_path": ckpt_path,
        "best_val_auc": best_auc,
        "history": history,
        "device": device,
        "decision_threshold": decision_threshold,
        "tensorboard_dir": tb_log_dir,
    }


def load_distinguisher(
    path: Path | str,
    *,
    device: str = "auto",
) -> CnnDistinguisher:
    import torch

    dev = _resolve_device(device)
    try:
        data = torch.load(path, map_location=dev, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This project requires a PyTorch version with safe weights_only "
            "checkpoint loading support. Upgrade PyTorch and retry."
        ) from exc
    if not isinstance(data, dict) or data.get("schema_version") != 2:
        raise ValueError("unsupported or malformed checkpoint schema")
    if not isinstance(data.get("state_dict"), dict):
        raise ValueError("checkpoint is missing a state_dict")
    train_config = data.get("train_config")
    if not isinstance(train_config, dict):
        raise ValueError("checkpoint is missing its training configuration")
    channels = tuple(train_config.get("channels", (32, 64, 128)))
    if len(channels) not in (2, 3) or any(
        isinstance(channel, bool) or not isinstance(channel, int) or channel < 1
        for channel in channels
    ):
        raise ValueError("checkpoint contains invalid CNN channels")
    threshold = float(data.get("decision_threshold", 0.5))
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("checkpoint contains an invalid decision threshold")
    model = build_model(channels).to(dev)
    model.load_state_dict(data["state_dict"])
    model.decision_threshold = threshold
    model.checkpoint_metadata = data.get("experiment", {})
    model.eval()
    return model
