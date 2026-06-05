"""Matplotlib-inspired figure objects for notebook use."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from beautiful_brains.exceptions import ConfigError, DependencyError

_VIDEO_EXTENSIONS = (".gif", ".png", ".apng", ".apgn", ".webp", ".mp4")
_DEFAULT_FIGURE_SIZE = 8.0


@dataclass
class BBPanel:
    layers: list[Any] = field(default_factory=list)

    def append(self, layer: Any) -> BBPanel:
        self.layers.append(_as_rgba_image(layer))
        return self

    def clear(self) -> None:
        self.layers.clear()

    def render(self, *, size_px: tuple[int, int], fit_scale: float | None = None):
        Image = _import_pillow()

        canvas = Image.new("RGBA", size_px, (0, 0, 0, 0))
        for layer in self.layers:
            fitted = _fit(layer, size_px, fit_scale=fit_scale)
            canvas.alpha_composite(fitted)
        return canvas


@dataclass
class BBFigure:
    size: float = _DEFAULT_FIGURE_SIZE
    grid: int | tuple[int, int] = (1, 1)
    dpi: int = 300
    background: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0, 255)
    panel_aspect: float = 1.0
    preserve_scale: bool = True
    box_ratio: InitVar[float | None] = None
    panels: dict[tuple[int, int], BBPanel] = field(default_factory=dict)

    def __post_init__(self, box_ratio: float | None) -> None:
        self.grid = _figure_grid(self.grid)
        self.size = _figure_size(self.size)
        self.dpi = _figure_dpi(self.dpi)
        if box_ratio is not None:
            if self.panel_aspect != 1.0:
                raise ValueError("Use panel_aspect or box_ratio, not both.")
            self.panel_aspect = box_ratio
        self.panel_aspect = _figure_panel_aspect(self.panel_aspect)
        self.preserve_scale = _figure_preserve_scale(self.preserve_scale)
        cols, rows = self.grid
        self.background = _rgba_color(self.background)
        for row in range(rows):
            for col in range(cols):
                self.panels[(col, row)] = BBPanel()

    def __repr__(self) -> str:
        total = len(self.panels)
        populated = sum(1 for panel in self.panels.values() if panel.layers)
        return (
            "BBFigure("
            f"size={self.size!r}, "
            f"grid={self.grid!r}, "
            f"dpi={self.dpi!r}, "
            f"panel_aspect={self.panel_aspect!r}, "
            f"preserve_scale={self.preserve_scale!r}, "
            f"background={self.background!r}, "
            f"panels={populated}/{total}"
            ")"
        )

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("BBFigure(...)" if cycle else repr(self))

    def __call__(self, col: int, row: int = 0) -> BBPanel:
        return self._panel(col, row)

    def __getitem__(self, key: int | tuple[int, int]) -> BBPanel:
        col, row = self._panel_key(key)
        return self._panel(col, row)

    def __setitem__(self, key: int | tuple[int, int], layer: Any) -> None:
        col, row = self._panel_key(key)
        panel = self._panel(col, row)
        panel.clear()
        panel.append(layer)

    def render(self):
        Image = _import_pillow()

        width_px, height_px = _render_size_px(
            self.size,
            grid=self.grid,
            dpi=self.dpi,
            panel_aspect=self.panel_aspect,
        )
        output = Image.new("RGBA", (width_px, height_px), self.background)
        panel_items = _panel_render_items(self, width_px=width_px, height_px=height_px)
        fit_scale = _shared_fit_scale(panel_items) if self.preserve_scale else None
        for _col, _row, (x0, y0, x1, y1), panel in panel_items:
            rendered_panel = panel.render(size_px=(x1 - x0, y1 - y0), fit_scale=fit_scale)
            output.alpha_composite(rendered_panel, (x0, y0))
        return output

    def save(self, path: str | Path):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        image = self.render()
        image.save(path)
        return path

    def _panel(self, col: int, row: int) -> BBPanel:
        cols, rows = self.grid
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise IndexError(f"Panel ({col}, {row}) is outside figure grid {self.grid}.")
        return self.panels[(col, row)]

    def _panel_key(self, key: int | tuple[int, int]) -> tuple[int, int]:
        cols, rows = self.grid
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Figure panel keys must be an index or a (column, row) pair.")
            return (_panel_index(key[0], name="column"), _panel_index(key[1], name="row"))
        index = _panel_index(key, name="index")
        if index < 0 or index >= cols * rows:
            raise IndexError(f"Panel index {index} is outside figure grid {self.grid}.")
        return (index % cols, index // cols)


def bbfigure(
    *,
    size: float = _DEFAULT_FIGURE_SIZE,
    grid: int | tuple[int, int] = (1, 1),
    dpi: int = 300,
    background: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0, 255),
    panel_aspect: float = 1.0,
    preserve_scale: bool = True,
    box_ratio: float | None = None,
) -> BBFigure:
    """Create a notebook-friendly figure frame.

    ``size`` is the diagonal length of the rendered image in inches.
    ``grid`` is formatted as ``(columns, rows)``.
    ``panel_aspect`` is the width/height ratio of each grid panel.
    ``preserve_scale`` keeps one shared visual scale across all panels.
    """

    return BBFigure(
        size=size,
        grid=grid,
        dpi=dpi,
        background=background,
        panel_aspect=panel_aspect,
        preserve_scale=preserve_scale,
        box_ratio=box_ratio,
    )


def make_video(frames: Iterable[Any], timing: float, filepath: str | Path) -> Path:
    """Save a sequence of BBFigure frames as an animation."""

    output = Path(filepath).expanduser()
    extension = output.suffix.lower()
    if extension not in _VIDEO_EXTENSIONS:
        raise ConfigError(
            "make_video() filepath must end with one of these extensions: "
            f"{', '.join(_VIDEO_EXTENSIONS)}."
        )

    duration_ms = _duration_ms(timing)
    prepared = _video_frames(frames)
    output.parent.mkdir(parents=True, exist_ok=True)

    if extension == ".mp4":
        _save_mp4(prepared, output, timing=float(timing))
    elif extension == ".gif":
        _save_pillow_animation(prepared, output, duration_ms=duration_ms, format_name="GIF")
    elif extension == ".webp":
        _save_pillow_animation(prepared, output, duration_ms=duration_ms, format_name="WEBP")
    else:
        _save_pillow_animation(prepared, output, duration_ms=duration_ms, format_name="PNG")
    return output


def _fit(image, size_px: tuple[int, int], *, fit_scale: float | None = None):
    Image = _import_pillow()
    resample = getattr(Image.Resampling, "LANCZOS", 1)
    width, height = size_px
    local_scale = min(width / image.width, height / image.height)
    scale = local_scale if fit_scale is None else min(fit_scale, local_scale)
    new_size = (max(int(image.width * scale), 1), max(int(image.height * scale), 1))
    resized = image.resize(new_size, resample=resample)
    canvas = Image.new("RGBA", size_px, (0, 0, 0, 0))
    canvas.alpha_composite(resized, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
    return canvas


def _as_rgba_image(layer, *, context: str = "Figure layers"):
    Image = _import_pillow()
    if hasattr(layer, "render"):
        layer = layer.render()
    if not isinstance(layer, Image.Image):
        raise TypeError(f"{context} must be PIL images or objects with a render() method.")
    if layer.mode != "RGBA":
        layer = layer.convert("RGBA")
    return layer.copy()


def _rgba_color(
    color: tuple[int, int, int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if len(color) == 3:
        rgba = (*color, 255)
    elif len(color) == 4:
        rgba = color
    else:
        raise ValueError("Figure background must be an RGB or RGBA tuple.")
    rgba = tuple(int(channel) for channel in rgba)
    if any(channel < 0 or channel > 255 for channel in rgba):
        raise ValueError("Figure background channels must be in the 0-255 range.")
    return rgba  # type: ignore[return-value]


def _figure_grid(grid: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(grid, int):
        cols, rows = grid, 1
    else:
        try:
            cols, rows = grid
        except (TypeError, ValueError) as exc:
            raise ValueError("Figure grid must be an integer or a (columns, rows) tuple.") from exc
    cols = _panel_index(cols, name="grid column count")
    rows = _panel_index(rows, name="grid row count")
    if cols <= 0 or rows <= 0:
        raise ValueError("Figure grid dimensions must be positive.")
    return (cols, rows)


def _figure_size(size: float) -> float:
    if isinstance(size, bool):
        raise ValueError("Figure size must be a positive number of inches.")
    try:
        value = float(size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Figure size must be a positive number of inches.") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Figure size must be a positive number of inches.")
    return value


def _figure_dpi(dpi: int) -> int:
    if isinstance(dpi, bool):
        raise ValueError("Figure dpi must be a positive integer.")
    try:
        value = int(dpi)
    except (TypeError, ValueError) as exc:
        raise ValueError("Figure dpi must be a positive integer.") from exc
    if value != dpi or value <= 0:
        raise ValueError("Figure dpi must be a positive integer.")
    return value


def _figure_panel_aspect(panel_aspect: float) -> float:
    if isinstance(panel_aspect, bool):
        raise ValueError("Figure panel_aspect must be a positive number.")
    try:
        value = float(panel_aspect)
    except (TypeError, ValueError) as exc:
        raise ValueError("Figure panel_aspect must be a positive number.") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Figure panel_aspect must be a positive number.")
    return value


def _figure_preserve_scale(preserve_scale: bool) -> bool:
    if not isinstance(preserve_scale, bool):
        raise ValueError("Figure preserve_scale must be True or False.")
    return preserve_scale


def _render_size_px(
    size: float,
    *,
    grid: tuple[int, int],
    dpi: int,
    panel_aspect: float,
) -> tuple[int, int]:
    cols, rows = grid
    width_units = cols * panel_aspect
    height_units = rows
    diagonal_units = sqrt(width_units**2 + height_units**2)
    width = size * width_units / diagonal_units
    height = size * height_units / diagonal_units
    return (max(int(round(width * dpi)), 1), max(int(round(height * dpi)), 1))


def _panel_render_items(
    figure: BBFigure,
    *,
    width_px: int,
    height_px: int,
) -> list[tuple[int, int, tuple[int, int, int, int], BBPanel]]:
    cols, rows = figure.grid
    items = []
    for row in range(rows):
        y0, y1 = _grid_bounds(row, rows, height_px)
        for col in range(cols):
            x0, x1 = _grid_bounds(col, cols, width_px)
            items.append((col, row, (x0, y0, x1, y1), figure._panel(col, row)))
    return items


def _shared_fit_scale(
    panel_items: list[tuple[int, int, tuple[int, int, int, int], BBPanel]],
) -> float | None:
    scales = []
    for _col, _row, (x0, y0, x1, y1), panel in panel_items:
        width = x1 - x0
        height = y1 - y0
        for layer in panel.layers:
            scales.append(min(width / layer.width, height / layer.height))
    return min(scales) if scales else None


def _grid_bounds(index: int, count: int, total_px: int) -> tuple[int, int]:
    start = int(round(index * total_px / count))
    stop = int(round((index + 1) * total_px / count))
    return (start, max(stop, start + 1))


def _panel_index(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise IndexError(f"Figure {name} must be an integer.")
    try:
        index = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise IndexError(f"Figure {name} must be an integer.") from exc
    if index != value:
        raise IndexError(f"Figure {name} must be an integer.")
    return index


def _import_pillow():
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "Pillow is required to render figures and make GIF, APNG, or WebP videos. "
            "Install it with `python -m pip install Pillow`."
        ) from exc
    return Image


def _duration_ms(timing: float) -> int:
    try:
        timing = float(timing)
    except (TypeError, ValueError) as exc:
        raise ConfigError("make_video() timing must be a positive number of seconds.") from exc
    if timing <= 0:
        raise ConfigError("make_video() timing must be a positive number of seconds.")
    return max(int(round(timing * 1000)), 1)


def _video_frames(frames: Iterable[Any]) -> list[Any]:
    if isinstance(frames, (str, bytes)):
        raise ConfigError("make_video() frames must be a non-empty iterable of BBFigure objects.")
    try:
        frame_list = list(frames)
    except TypeError as exc:
        raise ConfigError(
            "make_video() frames must be a non-empty iterable of BBFigure objects."
        ) from exc
    if not frame_list:
        raise ConfigError("make_video() frames must include at least one frame.")
    if any(not isinstance(frame, BBFigure) for frame in frame_list):
        raise ConfigError("make_video() frames must all be BBFigure objects.")

    images = [_as_rgba_image(frame.render()) for frame in frame_list]
    size = images[0].size
    if any(image.size != size for image in images):
        raise ConfigError("make_video() rendered figures must all have the same pixel dimensions.")
    return images


def _save_pillow_animation(
    frames: list[Any],
    filepath: Path,
    *,
    duration_ms: int,
    format_name: str,
) -> None:
    frames[0].save(
        filepath,
        format=format_name,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def _save_mp4(frames: list[Any], filepath: Path, *, timing: float) -> None:
    writer_factory = _imageio_writer_factory()
    fps = 1.0 / timing
    with writer_factory(filepath, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))


def _imageio_writer_factory():
    try:
        import imageio
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "make_video() requires imageio and imageio-ffmpeg to save MP4 files. "
            "Install them with `python -m pip install imageio imageio-ffmpeg`."
        ) from exc
    try:
        import imageio_ffmpeg  # noqa: F401
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "make_video() requires imageio-ffmpeg to save MP4 files. "
            "Install it with `python -m pip install imageio-ffmpeg`."
        ) from exc

    def factory(filepath: Path, *, fps: float):
        return imageio.get_writer(
            filepath,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
        )

    return factory
