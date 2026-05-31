from __future__ import annotations

import numpy as np
from scipy import ndimage


def fft_magnitude(image: np.ndarray, log: bool = True) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    img = img - np.mean(img)
    win_y = np.hanning(img.shape[0])[:, None]
    win_x = np.hanning(img.shape[1])[None, :]
    F = np.fft.fftshift(np.fft.fft2(img * win_y * win_x))
    mag = np.abs(F)
    return np.log1p(mag) if log else mag


def frequency_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    return np.meshgrid(fx, fy)


def detect_frequency_peaks(
    image: np.ndarray,
    top_n: int = 8,
    min_radius: int = 8,
    neighborhood: int = 9,
) -> list[dict[str, float]]:
    mag = fft_magnitude(image, log=False)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(mag.shape)
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    work = mag.copy()
    work[radius < min_radius] = 0.0
    local_max = work == ndimage.maximum_filter(work, size=neighborhood)
    coords = np.argwhere(local_max)
    if coords.size == 0:
        return []
    values = work[coords[:, 0], coords[:, 1]]
    order = np.argsort(values)[::-1]
    fx_grid, fy_grid = frequency_grid(mag.shape)
    peaks = []
    used: list[tuple[int, int]] = []
    for idx in order:
        y, x = int(coords[idx, 0]), int(coords[idx, 1])
        if values[idx] <= 0:
            continue
        if any((y - uy) ** 2 + (x - ux) ** 2 < neighborhood**2 for uy, ux in used):
            continue
        used.append((y, x))
        peaks.append(
            {
                "pixel_y": y,
                "pixel_x": x,
                "offset_y": y - cy,
                "offset_x": x - cx,
                "freq_y_cycles_per_px": float(fy_grid[y, x]),
                "freq_x_cycles_per_px": float(fx_grid[y, x]),
                "magnitude": float(values[idx]),
            }
        )
        if len(peaks) >= top_n:
            break
    return peaks


def circular_peak_mask(shape: tuple[int, int], peaks: list[dict[str, float]], radius: int = 4) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices(shape)
    mask = np.zeros(shape, dtype=bool)
    for p in peaks:
        y, x = int(p["pixel_y"]), int(p["pixel_x"])
        mask |= (yy - y) ** 2 + (xx - x) ** 2 <= radius * radius
    return mask


def masked_fft_energy(image: np.ndarray, mask: np.ndarray) -> float:
    mag = fft_magnitude(image, log=False)
    denom = np.linalg.norm(mag.ravel()) + 1e-12
    return float(np.linalg.norm(mag[mask].ravel()) / denom)


def stripe_orientation_from_peak(peak: dict[str, float]) -> str:
    fx = abs(peak["freq_x_cycles_per_px"])
    fy = abs(peak["freq_y_cycles_per_px"])
    if fx > fy:
        return "vertical_stripes_x_variation"
    if fy > fx:
        return "horizontal_stripes_y_variation"
    return "diagonal_or_uncertain"

