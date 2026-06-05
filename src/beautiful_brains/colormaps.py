"""Matplotlib colormap helpers."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from beautiful_brains.exceptions import ConfigError

_BUILTIN_COLORMAP_FILES: dict[str, str] = {
    "gray": "gray.csv",
    "grayscale": "gray.csv",
    "greyscale": "gray.csv",
    "grey": "gray.csv",
    "soft-gray": "soft-gray.csv",
    "soft-grayscale": "soft-gray.csv",
    "soft-greyscale": "soft-gray.csv",
    "soft-grey": "soft-gray.csv",
    "viridis": "viridis.csv",
    "soft-viridis": "soft-viridis.csv",
    "plasma": "plasma.csv",
    "soft-plasma": "soft-plasma.csv",
    "greys": "greys.csv",
    "soft-greys": "soft-greys.csv",
    "jet": "jet.csv",
    "soft-jet": "soft-jet.csv",
    "rainbow": "rainbow.csv",
    "soft-rainbow": "soft-rainbow.csv",
    "coolwarm": "coolwarm.csv",
    "soft-coolwarm": "soft-coolwarm.csv",
    "bwr": "bwr.csv",
    "soft-bwr": "soft-bwr.csv",
    "seismic": "seismic.csv",
    "soft-seismic": "soft-seismic.csv",
    "bone": "bone.csv",
    "soft-bone": "soft-bone.csv",
    "hot": "hot.csv",
    "soft-hot": "soft-hot.csv",
    "copper": "copper.csv",
    "soft-copper": "soft-copper.csv",
    "magma": "magma.csv",
    "soft-magma": "soft-magma.csv",
    "reds": "reds.csv",
    "soft-reds": "soft-reds.csv",
    "blues": "blues.csv",
    "soft-blues": "soft-blues.csv",
    "nih": "nih.csv",
    "soft-nih": "soft-nih.csv",
}


def _matplotlib():
    try:
        import matplotlib as mpl
        from matplotlib import colormaps
        from matplotlib.colors import Colormap, ListedColormap
    except ModuleNotFoundError as exc:
        raise ConfigError("matplotlib is required for colormap handling.") from exc
    return mpl, colormaps, Colormap, ListedColormap


def builtin_colormap_path(name: str) -> Path:
    normalized = name.lower()
    assets = files("beautiful_brains.assets.colormaps")
    if normalized in _BUILTIN_COLORMAP_FILES:
        return Path(str(assets.joinpath(_BUILTIN_COLORMAP_FILES[normalized])))
    raise ConfigError(f"Unknown built-in colormap: {name}")


def load_colormap(path_or_name: str | Path) -> np.ndarray:
    path = _resolve_colormap_path(path_or_name)
    colors = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    if colors.ndim == 1:
        colors = colors[None, :]
    if colors.ndim != 2 or colors.shape[1] not in {3, 4}:
        raise ConfigError("Colormap files must have 3 or 4 numeric columns.")
    if colors.shape[1] == 3:
        alpha = np.ones((colors.shape[0], 1), dtype=np.float32)
        colors = np.concatenate([colors, alpha], axis=1)
    if np.nanmax(colors) > 1:
        colors = colors / 255
    return np.clip(colors, 0, 1).astype(np.float32)


def as_listed_colormap(colormap: Any | None = None, *, samples: int = 256):
    """Return a matplotlib ``ListedColormap`` from a colormap object or name."""

    _, _, _, ListedColormap = _matplotlib()
    resolved = resolve_colormap(colormap)
    if isinstance(resolved, ListedColormap):
        return ListedColormap(_sample_colormap(resolved, samples), name=resolved.name)
    return ListedColormap(_sample_colormap(resolved, samples), name=resolved.name)


def resolve_colormap(colormap: Any | None = None):
    """Return a matplotlib colormap object from a colormap object, name, or color table."""

    _, colormaps, Colormap, ListedColormap = _matplotlib()
    if colormap is None:
        return colormaps["gray"]

    if isinstance(colormap, Colormap):
        return colormap
    if isinstance(colormap, Path):
        return ListedColormap(load_colormap(colormap), name=colormap.stem)
    if isinstance(colormap, str):
        file_path = _existing_path(colormap)
        if file_path is not None:
            return ListedColormap(load_colormap(file_path), name=file_path.stem)
        try:
            return colormaps[colormap]
        except KeyError:
            return ListedColormap(load_colormap(colormap), name=colormap)

    if isinstance(colormap, np.ndarray | list | tuple):
        colors = np.asarray(colormap, dtype=np.float32)
        if colors.ndim != 2 or colors.shape[1] not in {3, 4}:
            raise ConfigError("Colormap arrays must have shape (N, 3) or (N, 4).")
        if colors.shape[1] == 3:
            colors = np.concatenate([colors, np.ones((colors.shape[0], 1))], axis=1)
        if np.nanmax(colors) > 1:
            colors = colors / 255
        return ListedColormap(np.clip(colors, 0, 1), name="custom")

    raise ConfigError("colormap must be a matplotlib colormap, name, or color table.")


def _resolve_colormap_path(path_or_name: str | Path) -> Path:
    if isinstance(path_or_name, Path):
        path = path_or_name.expanduser()
        if path.exists():
            return path
        raise ConfigError(f"Colormap file does not exist: {path}")

    file_path = _existing_path(path_or_name)
    if file_path is not None:
        return file_path
    if _looks_like_path(path_or_name):
        raise ConfigError(f"Colormap file does not exist: {Path(path_or_name).expanduser()}")
    return builtin_colormap_path(path_or_name)


def _existing_path(value: str) -> Path | None:
    path = Path(value).expanduser()
    return path if path.exists() else None


def _looks_like_path(value: str) -> bool:
    path = Path(value)
    return path.parent != Path(".") or path.suffix != ""


def colormap_to_array(colormap: Any | None, *, samples: int = 256) -> np.ndarray:
    listed = as_listed_colormap(colormap, samples=samples)
    return _sample_colormap(listed, samples).astype(np.float32)


def colormap_from_array(colors: np.ndarray, *, name: str = "saved"):
    _, _, _, ListedColormap = _matplotlib()
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] not in {3, 4}:
        raise ConfigError("Saved colormap arrays must have shape (N, 3) or (N, 4).")
    if colors.shape[1] == 3:
        colors = np.concatenate([colors, np.ones((colors.shape[0], 1))], axis=1)
    if np.nanmax(colors) > 1:
        colors = colors / 255
    return ListedColormap(np.clip(colors, 0, 1), name=name)


def scalar_to_rgba(
    data: np.ndarray,
    *,
    colormap: Any | None,
    bounds: tuple[float, float] | None,
    alpha: float,
) -> np.ndarray:
    cmap = as_listed_colormap(colormap)
    finite = np.isfinite(data)
    if bounds is None:
        if np.any(finite):
            lo = float(np.nanmin(data[finite]))
            hi = float(np.nanmax(data[finite]))
        else:
            lo, hi = 0.0, 1.0
    else:
        lo, hi = bounds
    if hi <= lo:
        hi = lo + 1.0

    scaled = np.clip((data - lo) / (hi - lo), 0, 1)
    rgba = cmap(scaled, bytes=True)
    rgba[..., 3] = np.where(finite, np.clip(alpha, 0, 1) * rgba[..., 3], 0)
    return rgba.astype(np.uint8)


def colorbar_rgba(colormap: Any | None, width: int, height: int) -> np.ndarray:
    gradient = np.repeat(np.linspace(0, 1, width, dtype=np.float32)[None, :], height, axis=0)
    return scalar_to_rgba(gradient, colormap=colormap, bounds=(0, 1), alpha=1.0)


def colorbar(
    *,
    colormap: Any | None = None,
    height: int = 50,
    length: int = 400,
    border_size: int = 1,
    border_color: Sequence[int] = (0, 0, 0),
):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ConfigError("Pillow is required to render colorbars.") from exc

    if height <= 0 or length <= 0:
        raise ConfigError("Colorbar height and length must be positive.")
    if border_size < 0:
        raise ConfigError("Colorbar border_size must be non-negative.")
    border = tuple(int(channel) for channel in border_color)
    if len(border) != 3 or any(channel < 0 or channel > 255 for channel in border):
        raise ConfigError("border_color must contain three 0-255 RGB values.")

    rgba = colorbar_rgba(colormap, width=int(length), height=int(height))
    if border_size:
        size = min(int(border_size), rgba.shape[0] // 2, rgba.shape[1] // 2)
        if size:
            border_rgba = (*border, 255)
            rgba[:size, :, :] = border_rgba
            rgba[-size:, :, :] = border_rgba
            rgba[:, :size, :] = border_rgba
            rgba[:, -size:, :] = border_rgba
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")


def _sample_colormap(colormap: Any, samples: int) -> np.ndarray:
    samples = max(int(samples), 2)
    colors = colormap(np.linspace(0, 1, samples), bytes=False)
    return np.asarray(colors, dtype=np.float32)
