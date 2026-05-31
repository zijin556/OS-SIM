from __future__ import annotations

import numpy as np


def default_phases(m: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, m, endpoint=False, dtype=np.float32)


def demodulate_phase_stack(Y: np.ndarray, phases: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Demodulate Y[..., M, H, W] into D0, Dc, Ds and A.

    The M=3 path follows the formula specified in the project brief. Other M
    values use a least-squares sinusoid fit with phases in radians.
    """
    arr = np.asarray(Y, dtype=np.float32)
    if arr.ndim < 3:
        raise ValueError("Y must have at least phase,height,width dimensions")
    m = arr.shape[-3]
    if m == 3 and phases is None:
        i1, i2, i3 = arr[..., 0, :, :], arr[..., 1, :, :], arr[..., 2, :, :]
        d0 = (i1 + i2 + i3) / 3.0
        dc = (2.0 * i1 - i2 - i3) / 3.0
        ds = (i3 - i2) / np.sqrt(3.0)
        amp = np.sqrt(np.maximum(dc * dc + ds * ds, 0.0))
        return {"D0": d0, "Dc": dc, "Ds": ds, "A": amp}
    if phases is None:
        phases = default_phases(m)
    phases = np.asarray(phases, dtype=np.float32)
    if phases.shape[0] != m:
        raise ValueError(f"Expected {m} phases, got {phases.shape[0]}")
    design = np.stack([np.ones_like(phases), np.cos(phases), np.sin(phases)], axis=1)
    pinv = np.linalg.pinv(design).astype(np.float32)  # 3,M
    flat = arr.reshape(*arr.shape[:-3], m, -1)
    coeff = np.einsum("cm,...mn->...cn", pinv, flat, optimize=True)
    out_shape = arr.shape[:-3] + arr.shape[-2:]
    d0 = coeff[..., 0, :].reshape(out_shape)
    dc = coeff[..., 1, :].reshape(out_shape)
    ds = coeff[..., 2, :].reshape(out_shape)
    amp = np.sqrt(np.maximum(dc * dc + ds * ds, 0.0))
    return {"D0": d0, "Dc": dc, "Ds": ds, "A": amp}


def demodulate_multigroup(Y: np.ndarray) -> dict[str, np.ndarray]:
    """Demodulate Y[K,G,M,H,W] group by group."""
    arr = np.asarray(Y, dtype=np.float32)
    if arr.ndim != 5:
        raise ValueError("Y must have shape K,G,M,H,W")
    pieces = {"D0": [], "Dc": [], "Ds": [], "A": []}
    for g in range(arr.shape[1]):
        demod = demodulate_phase_stack(arr[:, g])
        for key in pieces:
            pieces[key].append(demod[key])
    return {key: np.stack(value, axis=1) for key, value in pieces.items()}


def fuse_hv(A_h: np.ndarray, A_v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum((A_h * A_h + A_v * A_v) / 2.0, 0.0)).astype(np.float32)


def modulation_residual(Y: np.ndarray, A: np.ndarray, M: np.ndarray) -> np.ndarray:
    y_tilde = Y - Y.mean(axis=-3, keepdims=True)
    return y_tilde - A[..., None, :, :] * M


def rms(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))

