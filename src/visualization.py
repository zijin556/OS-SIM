from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_for_display(img: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr[np.isfinite(arr)], [pmin, pmax])
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def decimate(img: np.ndarray, max_side: int = 768) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim < 2:
        return arr
    h, w = arr.shape[-2:]
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    return arr[..., ::step, ::step]


def save_montage(
    images: list[np.ndarray] | np.ndarray,
    path: str | Path,
    titles: list[str] | None = None,
    cols: int = 4,
    cmap: str = "gray",
    max_side: int = 512,
) -> None:
    arr = np.asarray(images)
    if arr.ndim == 2:
        arr = arr[None, ...]
    n = arr.shape[0]
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(normalize_for_display(decimate(arr[i], max_side=max_side)), cmap=cmap)
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=9)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_scalar_image(img: np.ndarray, path: str | Path, title: str | None = None, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(decimate(img), cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_profiles(A_stack: np.ndarray, z_values: np.ndarray, path: str | Path, title: str = "Axis profiles") -> None:
    A = np.asarray(A_stack, dtype=np.float32)
    h, w = A.shape[-2:]
    points = [(h // 2, w // 2), (h // 3, w // 3), (2 * h // 3, 2 * w // 3)]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for y, x in points:
        ax.plot(z_values, A[:, y, x], marker="o", ms=2, label=f"({x},{y})")
    ax.set_xlabel("z")
    ax.set_ylabel("A")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_loss_curve(losses: list[float] | np.ndarray, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(np.asarray(losses), lw=1.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Latent optimization loss")
    ax.grid(True, alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

