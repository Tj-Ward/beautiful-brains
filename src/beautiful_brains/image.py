"""Notebook-first image loading, transforms, smoothing, and slice generation."""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from beautiful_brains.alignment import (
    AntsTransformFile,
    CropInfo,
    _debug_print,
    apply_ants_transform,
    bias_correct_volume,
    create_brainmask_array,
    create_mask_array,
    crop_volume,
    normalize_ants_interpolator,
    normalize_ants_warp,
    register_ants_transform,
    validate_crop_info,
)
from beautiful_brains.colormaps import (
    as_listed_colormap,
    colormap_from_array,
    colormap_to_array,
    resolve_colormap,
    scalar_to_rgba,
)
from beautiful_brains.exceptions import ConfigError
from beautiful_brains.io import (
    Volume,
    load_volume,
    smooth_fwhm,
    volume_from_data,
    volume_from_image,
)
from beautiful_brains.luts import load_lookup_table, resolve_lookup_table_path
from beautiful_brains.models import Plane
from beautiful_brains.utils import ensure_path, parse_labels

ImageKind = Literal["intensity", "delineation"]
AXIS_BY_PLANE: dict[Plane, int] = {"sagittal": 0, "coronal": 1, "axial": 2}
EDITABLE_BBIMAGE_FIELDS = {
    "scale",
    "interp",
    "sharpen",
    "colormap",
    "LUT",
    "threshold",
    "crop_info",
    "mask",
}


@dataclass(frozen=True)
class TransformRecord:
    """ANTs transform metadata that can be reused by another image."""

    warp: str | None = None
    template: Path | None = None
    forward_transforms: tuple[AntsTransformFile, ...] = field(default_factory=tuple)
    inverse_transforms: tuple[AntsTransformFile, ...] = field(default_factory=tuple)
    applied: bool = False
    template_volume: Volume | None = field(default=None, repr=False, compare=False)
    template_shape: tuple[int, ...] | None = None
    template_affine: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.template is not None:
            object.__setattr__(self, "template", Path(self.template))
        object.__setattr__(self, "forward_transforms", tuple(self.forward_transforms))
        object.__setattr__(self, "inverse_transforms", tuple(self.inverse_transforms))
        if self.template_shape is None and self.template_volume is not None:
            object.__setattr__(self, "template_shape", tuple(self.template_volume.data.shape))
        elif self.template_shape is not None:
            object.__setattr__(
                self,
                "template_shape",
                tuple(int(value) for value in self.template_shape),
            )
        if self.template_affine is None and self.template_volume is not None:
            object.__setattr__(
                self,
                "template_affine",
                np.asarray(self.template_volume.affine, dtype=np.float64),
            )
        elif self.template_affine is not None:
            object.__setattr__(
                self,
                "template_affine",
                np.asarray(self.template_affine, dtype=np.float64),
            )
        object.__setattr__(self, "applied", bool(self.applied))


@dataclass
class BBImage:
    """A loaded NIfTI volume prepared for notebook figure creation."""

    path: Path
    volume: Volume
    LUT: Path | None = None
    colormap: Any | None = None
    indices: tuple[int, ...] | None = None
    threshold: tuple[float, float] | None = None
    crop_info: CropInfo | None = None
    mask: np.ndarray | None = None
    transform_info: TransformRecord | None = None
    name: str | None = None
    kind: ImageKind = "intensity"
    scale: float = 3.0
    interp: str = "BICUBIC"
    sharpen: int = 0
    _bb_initialized: bool = field(default=False, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if not getattr(self, "_bb_initialized", False):
            object.__setattr__(self, name, value)
            return
        if name not in EDITABLE_BBIMAGE_FIELDS:
            raise AttributeError(
                "Only display metadata fields and mask are editable: "
                "scale, interp, sharpen, colormap, LUT, threshold, crop_info, mask."
            )
        object.__setattr__(self, name, _coerce_editable_field(self, name=name, value=value))

    def __post_init__(self) -> None:
        object.__setattr__(self, "_bb_initialized", False)
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "kind", _validate_kind(self.kind))
        object.__setattr__(self, "LUT", Path(self.LUT) if self.LUT is not None else None)
        if self.kind == "intensity" and self.LUT is not None:
            raise ConfigError("intensity images use colormap, not LUT.")
        if self.kind == "delineation" and self.colormap is not None:
            raise ConfigError("delineation images use LUT, not colormap.")
        object.__setattr__(
            self,
            "threshold",
            _validate_threshold(self.threshold) if self.kind == "intensity" else None,
        )
        object.__setattr__(
            self,
            "crop_info",
            validate_crop_info(self.crop_info, shape=self.volume.data.shape),
        )
        object.__setattr__(
            self,
            "colormap",
            resolve_colormap(self.colormap) if self.kind == "intensity" else None,
        )
        object.__setattr__(self, "name", self.name or Path(self.path).name)
        object.__setattr__(
            self,
            "volume",
            Volume(
                path=self.volume.path,
                image=self.volume.image,
                data=_readonly_array(self.volume.data),
                affine=_readonly_array(self.volume.affine),
            ),
        )
        object.__setattr__(
            self,
            "mask",
            _normalize_mask(self.mask, shape=self.volume.data.shape),
        )
        object.__setattr__(self, "scale", _validate_scale(self.scale))
        resolved_interp = _normalize_interp_name(self.interp)
        if self.kind == "delineation":
            resolved_interp = "NEAREST"
        object.__setattr__(self, "interp", resolved_interp)
        object.__setattr__(self, "sharpen", _validate_sharpen(self.sharpen))
        if self.kind == "delineation" and self.sharpen:
            raise ConfigError("sharpen is only supported for intensity images.")
        object.__setattr__(self, "_bb_initialized", True)

    def __repr__(self) -> str:
        colormap_name = getattr(self.colormap, "name", None) if self.colormap is not None else None
        LUT_name = self.LUT.name if self.LUT is not None else None
        mask = "set" if self.mask is not None else "none"
        crop = "set" if self.crop_info is not None else "none"
        transform = _transform_status(self.transform_info)
        return (
            "BBImage("
            f"name={self.name!r}, "
            f"kind={self.kind!r}, "
            f"shape={self.volume.data.shape}, "
            f"dtype={self.volume.data.dtype}, "
            f"colormap={colormap_name!r}, "
            f"LUT={LUT_name!r}, "
            f"threshold={self.threshold!r}, "
            f"mask={mask!r}, "
            f"crop_info={crop!r}, "
            f"transform_info={transform!r}, "
            f"scale={self.scale!r}, "
            f"interp={self.interp!r}, "
            f"sharpen={self.sharpen!r}"
            ")"
        )

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("BBImage(...)" if cycle else repr(self))

    def slice(
        self,
        index: int,
        plane: Plane = "axial",
        *,
        interp: Any | None = None,
        alpha: float | None = None,
        outline: bool | int | None = None,
    ):
        return self.render_slice(
            index,
            plane=plane,
            interp=interp,
            alpha=alpha,
            outline=outline,
        )

    def render_slice(
        self,
        index: int,
        *,
        plane: Plane = "axial",
        interp: Any | None = None,
        alpha: float | None = None,
        outline: bool | int | None = None,
    ):
        source = self.reslice() if _has_pending_transform(self.transform_info) else self
        axis = AXIS_BY_PLANE[plane]
        shape = _display_volume_shape(source)
        if index < 0 or index >= shape[axis]:
            raise IndexError(
                f"Slice {index} is outside {plane} axis with size {shape[axis]}."
            )
        slice_rgba = _render_slice_rgba(source, axis=axis, index=index, interp=interp)
        if outline:
            slice_rgba = _outline_slice(slice_rgba, width=1 if outline is True else int(outline))
        slice_rgba = _multiply_alpha(slice_rgba, 1.0 if alpha is None else alpha)
        return _pil_from_slice(slice_rgba)

    def reslice(
        self,
        interp: Any | None = None,
        *,
        cache_dir: str | Path | None = None,
        debug: bool = False,
    ) -> BBImage:
        """Apply a pending spatial transform and return a resliced image."""

        return _reslice_image(
            self,
            interp=interp,
            cache_dir=cache_dir,
            debug=debug,
        )

    def metadata(self) -> None:
        """Print this image's metadata without dumping matrices."""

        print(_metadata_text(self))

    def _with_data(
        self,
        *,
        data: np.ndarray,
        path: Path | None = None,
        volume: Volume | None = None,
        transform_info: TransformRecord | None | Literal["keep"] = "keep",
        crop_info: CropInfo | None | Literal["keep"] = "keep",
        mask: np.ndarray | None | Literal["keep"] = "keep",
    ) -> BBImage:
        volume = volume or Volume(
            path=self.volume.path,
            image=self.volume.image,
            data=data,
            affine=self.volume.affine,
        )
        resolved_mask = self.mask if isinstance(mask, str) and mask == "keep" else mask
        return BBImage(
            path=path or self.path,
            volume=volume,
            LUT=self.LUT,
            colormap=self.colormap,
            indices=self.indices,
            threshold=self.threshold,
            crop_info=self.crop_info if crop_info == "keep" else crop_info,
            mask=resolved_mask,
            transform_info=self.transform_info if transform_info == "keep" else transform_info,
            name=self.name,
            kind=self.kind,
            scale=self.scale,
            interp=self.interp,
            sharpen=self.sharpen,
        )

    def save(self, path: str | Path) -> Path:
        """Save this Beautiful-Brains image object to a ``.bbi`` archive."""

        output = _bbi_path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "beautiful-brains-image",
            "version": 1,
            "path": str(self.path),
            "LUT": str(self.LUT) if self.LUT is not None else None,
            "colormap_name": getattr(self.colormap, "name", None) if self.colormap else None,
            "indices": self.indices,
            "threshold": self.threshold,
            "crop_info": self.crop_info,
            "name": self.name,
            "kind": self.kind,
            "transform_info": _transform_metadata(self.transform_info),
            "scale": self.scale,
            "interp": self.interp,
            "sharpen": self.sharpen,
        }
        colormap_rgba = (
            np.empty((0, 4), dtype=np.float32)
            if self.colormap is None
            else colormap_to_array(self.colormap)
        )
        with output.open("wb") as handle:
            np.savez_compressed(
                handle,
                data=self.volume.data,
                mask=(
                    np.empty((0,), dtype=np.float16)
                    if self.mask is None
                    else np.asarray(self.mask, dtype=np.float16)
                ),
                affine=self.volume.affine,
                colormap_rgba=colormap_rgba,
                metadata=np.array(json.dumps(metadata)),
            )
        return output


def transform(
    *,
    source: BBImage,
    target: str | Path | BBImage,
    warp: str | None = None,
    cache_dir: str | Path | None = None,
    return_transform: bool = False,
    debug: bool = False,
) -> BBImage | TransformRecord:
    """Register ``source`` to ``target`` with ANTs without resampling voxel data."""

    if not isinstance(source, BBImage):
        raise ConfigError("transform() source must be a BBImage.")
    resolved_warp = normalize_ants_warp(warp)
    fixed_volume = _target_volume(target)
    result = register_ants_transform(
        fixed=fixed_volume,
        moving=source.volume,
        warp=resolved_warp,
        cache_dir=cache_dir,
        debug=debug,
    )
    transform_info = TransformRecord(
        warp=resolved_warp,
        template=_target_template(target),
        forward_transforms=result.forward_transforms,
        inverse_transforms=result.inverse_transforms,
        applied=False,
        template_volume=fixed_volume,
    )
    _validate_transform_files(transform_info)
    if return_transform:
        return transform_info
    return _image_with_pending_transform(source, transform_info=transform_info)


def apply_transform(
    *,
    source: BBImage,
    matrix: TransformRecord | BBImage,
) -> BBImage:
    """Attach a saved ANTs transform to ``source`` without resampling voxel data."""

    if not isinstance(source, BBImage):
        raise ConfigError("apply_transform() source must be a BBImage.")
    transform_info = _pending_transform_record(
        _as_transform_record(matrix),
    )
    return _image_with_pending_transform(source, transform_info=transform_info)


def crop(
    source: BBImage,
    *,
    how: str = "mean",
    pad: int = 0,
    crop_info: Sequence[Sequence[int]] | None = None,
    cache_dir: str | Path | None = None,
    debug: bool = False,
) -> BBImage:
    """Return a cropped copy of ``source`` with crop metadata preserved."""

    if not isinstance(source, BBImage):
        raise ConfigError("crop() source must be a BBImage.")
    working = (
        source.reslice(cache_dir=cache_dir, debug=debug)
        if _has_pending_transform(source.transform_info)
        else source
    )
    result = crop_volume(
        working.volume,
        how=how,
        pad=pad,
        crop_info=crop_info,
        cache_dir=cache_dir,
        debug=debug,
    )
    return working._with_data(
        data=working.volume.data,
        volume=working.volume,
        crop_info=result.crop_info,
        mask="keep",
    )


def create_mask(
    *,
    source: BBImage,
    pad: int = 0,
    low_thresh: float | None = None,
    max_thresh: float | None = None,
    cleanup: int | None = None,
    enhance: bool = True,
    cache_dir: str | Path | None = None,
    debug: bool = False,
) -> np.ndarray:
    """Create a float16 brain mask from a Beautiful-Brains image."""

    if not isinstance(source, BBImage):
        raise ConfigError("create_mask() source must be a BBImage.")
    mask = _validate_mask(
        create_mask_array(
            source.volume,
            pad=pad,
            low_thresh=low_thresh,
            max_thresh=max_thresh,
            cleanup=cleanup,
            cache_dir=cache_dir,
            debug=debug,
        ),
        shape=source.volume.data.shape,
    )
    if enhance:
        mask = _enhance_mask(mask, affine=source.volume.affine)
    return mask.astype(np.float16, copy=True)


def create_brainmask(
    *,
    source: BBImage,
    modality: str = "t1",
    threshold: float = 0.5,
    pad: int = 0,
    cleanup: int = 0,
    cache_dir: str | Path | None = None,
    debug: bool = False,
) -> np.ndarray:
    """Create a float16 brain-only skull-stripping mask from a Beautiful-Brains image."""

    if not isinstance(source, BBImage):
        raise ConfigError("create_brainmask() source must be a BBImage.")
    return _validate_mask(
        create_brainmask_array(
            source.volume,
            modality=modality,
            threshold=threshold,
            pad=pad,
            cleanup=cleanup,
            cache_dir=cache_dir,
            debug=debug,
        ),
        shape=source.volume.data.shape,
    ).astype(np.float16, copy=True)


def smooth_image(*, source: BBImage, kernel: float | Sequence[float]) -> BBImage:
    """Return a copy of ``source`` smoothed with an FWHM kernel in millimeters."""

    if not isinstance(source, BBImage):
        raise ConfigError("smooth_image() source must be a BBImage.")
    if source.kind != "intensity":
        raise ConfigError("smooth_image() is only supported for scalar intensity images.")
    fwhm = _smooth_kernel_arg(kernel, ndim=source.volume.data.ndim, name="smooth_image()")
    smoothed = smooth_fwhm(source.volume.data, source.volume.affine, fwhm).astype(np.float32)
    volume = Volume(
        path=source.volume.path,
        image=source.volume.image,
        data=smoothed,
        affine=source.volume.affine,
    )
    return source._with_data(data=smoothed, volume=volume)


def smooth_mask(
    source: np.ndarray | None = None,
    *,
    mask: np.ndarray | None = None,
    kernel: float | Sequence[float],
    affine: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth a 3D mask matrix and return a clipped float16 mask."""

    if source is not None and mask is not None:
        raise ConfigError("smooth_mask() accepts either source= or mask=, not both.")
    mask_array = np.asarray(mask if mask is not None else source)
    if source is None and mask is None:
        raise ConfigError("smooth_mask() requires a mask matrix.")
    if mask_array.ndim != 3:
        raise ConfigError("smooth_mask() source must be a 3D mask matrix.")
    validated = _validate_mask(mask_array, shape=mask_array.shape).astype(np.float32)
    resolved_affine = _mask_affine(affine)
    fwhm = _smooth_kernel_arg(kernel, ndim=validated.ndim, name="smooth_mask()")
    smoothed = smooth_fwhm(validated, resolved_affine, fwhm)
    smoothed = np.nan_to_num(smoothed, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(smoothed, 0, 1).astype(np.float16, copy=True)


def load(
    path: str | Path,
    *,
    LUT: str | Path | None = None,
    colormap: Any | None = None,
    threshold: tuple[float, float] | None = None,
    indices: Sequence[int] | str | None = None,
    scale: float | None = None,
    interp: Any | None = None,
    sharpen: int | None = None,
    bias_correction: bool = False,
    debug: bool = False,
) -> BBImage:
    """Load a NIfTI image and store slice rendering metadata."""

    path = ensure_path(path)
    if path.suffix == ".bbi":
        _validate_bbi_load_options(
            LUT=LUT,
            colormap=colormap,
            threshold=threshold,
            indices=indices,
            scale=scale,
            interp=interp,
            sharpen=sharpen,
            bias_correction=bias_correction,
        )
        return _load_bbi(path, debug=debug)

    _debug_print(debug, f"loading image: {path}")
    volume = load_volume(path)
    _debug_print(debug, f"loaded image shape: {volume.data.shape}")
    parsed_indices = _parse_indices(indices)
    if LUT is not None and threshold is not None:
        raise ConfigError("threshold can only be used when loading intensity images.")
    kind = _infer_kind(LUT_value=LUT, colormap=colormap, indices=parsed_indices)
    _debug_print(debug, f"image kind: {kind}")
    if bias_correction:
        if kind != "intensity":
            raise ConfigError("bias_correction=True is only supported for scalar intensity images.")
        volume = bias_correct_volume(volume, debug=debug)
    LUT_path = resolve_lookup_table_path(LUT) if kind == "delineation" else None
    resolved_colormap = resolve_colormap(colormap) if kind == "intensity" else None
    data = _prepare_data(
        volume.data,
        kind=kind,
        indices=parsed_indices,
    )
    if kind == "intensity" and threshold is None:
        threshold = _auto_image_threshold(data)
        _debug_print(debug, f"auto threshold: {threshold}")
    elif threshold is not None and kind == "intensity":
        _debug_print(debug, f"threshold: {threshold}")
    if kind == "intensity":
        _debug_print(debug, f"colormap: {getattr(resolved_colormap, 'name', 'custom')}")
    else:
        _debug_print(debug, f"LUT: {LUT_path}")
    resolved_scale = _validate_scale(3.0 if scale is None else scale)
    resolved_interp = _normalize_interp_name("BICUBIC" if interp is None else interp)
    resolved_sharpen = _validate_sharpen(0 if sharpen is None else sharpen)
    return BBImage(
        path=path,
        volume=Volume(path=volume.path, image=volume.image, data=data, affine=volume.affine),
        LUT=LUT_path,
        colormap=resolved_colormap,
        indices=parsed_indices,
        threshold=threshold,
        name=path.name,
        kind=kind,
        scale=resolved_scale,
        interp=resolved_interp,
        sharpen=resolved_sharpen,
    )


def _load_bbi(path: Path, *, debug: bool = False) -> BBImage:
    _debug_print(debug, f"loading Beautiful-Brains image archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format") != "beautiful-brains-image":
            raise ConfigError(f"Unsupported BBI archive format: {path}")
        data = np.asarray(archive["data"], dtype=np.float32)
        affine = np.asarray(archive["affine"], dtype=np.float64)
        mask = np.asarray(archive["mask"], dtype=np.float16)
        if "colormap_rgba" in archive.files:
            colormap_rgba = np.asarray(archive["colormap_rgba"], dtype=np.float32)
        else:
            colormap_rgba = np.empty((0, 4), dtype=np.float32)

    kind = _validate_kind(metadata.get("kind", "intensity"))
    transform_info = _transform_from_metadata(
        metadata.get("transform_info") or {},
    )
    volume = volume_from_data(path, data, affine)
    LUT_raw = metadata.get("LUT")
    LUT = Path(LUT_raw) if LUT_raw else None
    mask = None if mask.size == 0 else mask
    colormap = (
        colormap_from_array(colormap_rgba, name=metadata.get("colormap_name") or "saved")
        if colormap_rgba.size
        else (as_listed_colormap("gray") if kind == "intensity" else None)
    )
    indices = _optional_tuple(metadata.get("indices"))
    threshold = _optional_tuple(metadata.get("threshold"))
    crop_info = validate_crop_info(metadata.get("crop_info"), shape=data.shape)
    _debug_print(debug, f"loaded BBI shape: {data.shape}")
    return BBImage(
        path=path,
        volume=volume,
        LUT=LUT,
        colormap=colormap,
        indices=indices,
        threshold=threshold,
        crop_info=crop_info,
        mask=mask,
        transform_info=transform_info,
        name=metadata.get("name") or path.name,
        kind=kind,
        scale=metadata.get("scale", 3.0),
        interp=metadata.get("interp", "BICUBIC"),
        sharpen=metadata.get("sharpen", 0),
    )


def _prepare_data(
    data: np.ndarray,
    *,
    kind: ImageKind,
    indices: tuple[int, ...] | None,
) -> np.ndarray:
    if kind == "delineation":
        labels = np.rint(data).astype(np.float32)
        if indices is not None:
            labels = np.where(
                np.isin(labels.astype(np.int64), np.array(indices, dtype=np.int64)),
                labels,
                0,
            )
        return labels.astype(np.float32)
    return np.asarray(data, dtype=np.float32)


def _infer_kind(
    *,
    LUT_value: str | Path | None,
    colormap: Any | None,
    indices: tuple[int, ...] | None,
) -> ImageKind:
    if LUT_value is not None and colormap is not None:
        raise ConfigError("LUT and colormap cannot both be specified.")
    if LUT_value is not None:
        return "delineation"
    if indices is not None:
        raise ConfigError("indices can only be used when loading a delineation image with LUT=.")
    return "intensity"


def _parse_indices(value: Sequence[int] | str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return parse_labels(value)
    return tuple(int(item) for item in value)


def _render_slice_rgba(
    source: BBImage,
    *,
    axis: int,
    index: int,
    interp: Any | None = None,
) -> np.ndarray:
    display_interp = _slice_display_interp(source, interp)
    data_slice = _take_display_slice(
        source.volume.data,
        axis=axis,
        index=index,
        crop_info=source.crop_info,
    )
    output_shape = _slice_display_shape(source, axis=axis)
    display_data = _display_slice_data(
        data_slice,
        kind=source.kind,
        threshold=source.threshold,
        output_shape=output_shape,
        scale=source.scale,
        interp=display_interp,
        sharpen=source.sharpen,
    )
    display_mask = None
    if source.mask is not None:
        mask_slice = _take_display_slice(
            source.mask,
            axis=axis,
            index=index,
            crop_info=source.crop_info,
        )
        display_mask = _display_slice_mask(
            mask_slice,
            output_shape=output_shape,
            interp=display_interp,
        )
    threshold = (0.0, 1.0) if source.kind == "intensity" else source.threshold
    return _rgba_from_data(
        display_data,
        kind=source.kind,
        LUT=source.LUT,
        colormap=source.colormap,
        threshold=threshold,
        mask=display_mask,
    )


def _display_slice_data(
    data: np.ndarray,
    *,
    kind: ImageKind,
    threshold: tuple[float, float] | None,
    output_shape: tuple[int, int],
    scale: float,
    interp: str,
    sharpen: int,
) -> np.ndarray:
    display_data = np.array(data, dtype=np.float32, copy=True)
    if kind == "delineation":
        display_data = np.rint(display_data).astype(np.float32)
        display_data = _resize_2d_array(display_data, output_shape=output_shape, interp="NEAREST")
        return np.rint(display_data).astype(np.float32)
    display_data = _normalize_intensity_slice(display_data, bounds=threshold)
    display_data = _resize_2d_array(display_data, output_shape=output_shape, interp=interp)
    if sharpen:
        display_data = _sharpen_2d_data(display_data, sharpen=sharpen, scale=scale)
        display_data = np.clip(display_data, 0, 1)
    return display_data.astype(np.float32, copy=False)


def _display_slice_mask(
    mask: np.ndarray,
    *,
    output_shape: tuple[int, int],
    interp: str,
) -> np.ndarray:
    display_mask = np.array(mask, dtype=np.float32, copy=True)
    display_mask = _resize_2d_array(display_mask, output_shape=output_shape, interp=interp)
    display_mask = np.nan_to_num(display_mask, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(display_mask, 0, 1).astype(np.float16, copy=False)


def _slice_display_interp(source: BBImage, interp: Any | None) -> str:
    resolved = source.interp if interp is None else _normalize_interp_name(interp)
    return "NEAREST" if source.kind == "delineation" else resolved


def _resize_2d_array(
    array: np.ndarray,
    *,
    output_shape: tuple[int, int],
    interp: str,
) -> np.ndarray:
    resolved_shape = _validate_output_shape(output_shape)
    if tuple(array.shape) == resolved_shape:
        return np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ConfigError("slice rendering expects a 2D matrix.")
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ConfigError("Pillow is required for slice interpolation.") from exc
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    size = (resolved_shape[1], resolved_shape[0])
    resized = image.resize(size, resample=_pil_interp_filter(interp))
    return np.asarray(resized, dtype=np.float32)


def _normalize_intensity_slice(
    data: np.ndarray,
    *,
    bounds: tuple[float, float] | None,
) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, np.nan, dtype=np.float32)
    if bounds is None:
        lo = float(np.nanmin(values[finite]))
        hi = float(np.nanmax(values[finite]))
    else:
        lo, hi = bounds
    if hi <= lo:
        hi = lo + 1.0
    output = np.clip((values - lo) / (hi - lo), 0, 1).astype(np.float32)
    output[~finite] = np.nan
    return output


def _sharpen_2d_data(data: np.ndarray, *, sharpen: int, scale: float) -> np.ndarray:
    percent = _validate_sharpen(sharpen)
    values = np.asarray(data, dtype=np.float32)
    if percent == 0:
        return values
    finite = np.isfinite(values)
    filled = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    filled = np.clip(filled, 0, 1)
    try:
        from PIL import Image, ImageFilter
    except ModuleNotFoundError as exc:
        raise ConfigError("Pillow is required for slice sharpening.") from exc
    image = Image.fromarray(np.rint(filled * 255).astype(np.uint8), mode="L")
    sharpened = image.filter(
        ImageFilter.UnsharpMask(
            radius=_unsharp_mask_radius(scale),
            percent=percent,
            threshold=_unsharp_mask_threshold(scale),
        )
    )
    output = np.asarray(sharpened, dtype=np.float32) / 255.0
    output[~finite] = np.nan
    return output.astype(np.float32)


def _rgba_from_data(
    data: np.ndarray,
    *,
    kind: ImageKind,
    LUT: Path | None,
    colormap: Any | None,
    threshold: tuple[float, float] | None,
    mask: np.ndarray | None,
) -> np.ndarray:
    if kind == "intensity":
        rgba = scalar_to_rgba(data, colormap=colormap, bounds=threshold, alpha=1.0)
        return _apply_mask_to_rgba(rgba, mask)
    if LUT is None:
        raise ConfigError("Delineation images require a lookup table.")
    return _apply_mask_to_rgba(_label_rgba(data, LUT, alpha=1.0), mask)


def _auto_image_threshold(data: np.ndarray) -> tuple[float, float]:
    values = np.asarray(data)
    lower, upper = np.percentile(values, (2.5, 97.5))
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (0.0, 1.0)
        lower, upper = np.percentile(finite, (2.5, 97.5))
        lower = float(lower)
        upper = float(upper)
    if upper <= lower:
        upper = lower + 1.0
    return (lower, upper)


def _coerce_editable_field(image: BBImage, *, name: str, value: Any) -> Any:
    if name == "scale":
        return _validate_scale(value)
    if name == "interp":
        if image.kind == "delineation":
            return "NEAREST"
        return _normalize_interp_name(value)
    if name == "sharpen":
        resolved = _validate_sharpen(value)
        if image.kind == "delineation" and resolved:
            raise ConfigError("sharpen is only supported for intensity images.")
        return resolved
    if name == "colormap":
        if image.kind != "intensity":
            raise ConfigError("delineation images use LUT, not colormap.")
        return resolve_colormap(value)
    if name == "LUT":
        if image.kind != "delineation":
            if value is not None:
                raise ConfigError("intensity images use colormap, not LUT.")
            return None
        return resolve_lookup_table_path(value) if value is not None else None
    if name == "threshold":
        if image.kind != "intensity":
            if value is not None:
                raise ConfigError("threshold is only supported for intensity images.")
            return None
        return _validate_threshold(value)
    if name == "crop_info":
        return validate_crop_info(value, shape=_crop_info_reference_shape(image))
    if name == "mask":
        return _normalize_mask(value, shape=image.volume.data.shape)
    raise AttributeError(f"{name} is not editable.")


def _crop_info_reference_shape(image: BBImage) -> tuple[int, ...]:
    if _has_pending_transform(image.transform_info):
        return _template_volume_from_record(image.transform_info).data.shape
    return image.volume.data.shape


def _validate_threshold(value: tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        lo_raw, hi_raw = value
    except (TypeError, ValueError) as exc:
        raise ConfigError("threshold must be a two-value tuple: (min, max).") from exc
    lo = float(lo_raw)
    hi = float(hi_raw)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ConfigError("threshold values must be finite.")
    if hi <= lo:
        raise ConfigError("threshold max must be greater than min.")
    return (lo, hi)


def _coerce_transformed_data(data: np.ndarray, *, kind: ImageKind) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32).copy()
    if kind == "intensity":
        data[~np.isfinite(data)] = 0
        return data.astype(np.float32)
    return np.rint(data).astype(np.float32)


def _normalize_mask(mask: np.ndarray | None, *, shape: tuple[int, ...]) -> np.ndarray | None:
    if mask is None:
        return None
    return _validate_mask(mask, shape=shape).astype(np.float16, copy=True)


def _validate_mask(mask: np.ndarray, *, shape: tuple[int, ...]) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.shape != shape:
        raise ConfigError("mask must have the same shape as the image data.")
    if not np.all(np.isfinite(mask_array)):
        raise ConfigError("mask values must be finite.")
    if np.any((mask_array < 0) | (mask_array > 1)):
        raise ConfigError("mask values must be between 0 and 1.")
    return mask_array


def _apply_mask_to_rgba(rgba: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    normalized_mask = _normalize_mask(mask, shape=rgba.shape[:-1])
    if normalized_mask is None:
        return rgba
    output = np.asarray(rgba, dtype=np.uint8).copy()
    output[..., 3] = np.clip(
        output[..., 3].astype(np.float32) * normalized_mask.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return output


def _enhance_mask(mask: np.ndarray, *, affine: np.ndarray) -> np.ndarray:
    ndimage = _import_scipy_ndimage()
    filled = ndimage.binary_fill_holes(np.asarray(mask) >= 0.5)
    smoothed = smooth_fwhm(filled.astype(np.float32), affine, 4)
    smoothed = np.nan_to_num(smoothed, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(smoothed, 0, 1).astype(np.float16, copy=True)


def _smooth_kernel_arg(
    kernel: float | Sequence[float],
    *,
    ndim: int,
    name: str,
) -> float | tuple[float, ...]:
    if isinstance(kernel, int | float):
        value = float(kernel)
        if value < 0:
            raise ConfigError(f"{name} kernel values must be non-negative.")
        return value
    values = tuple(float(value) for value in kernel)
    if len(values) != ndim:
        raise ConfigError(f"{name} kernel length must match image dimensions ({ndim}).")
    if any(value < 0 for value in values):
        raise ConfigError(f"{name} kernel values must be non-negative.")
    return values


def _mask_affine(affine: np.ndarray | None) -> np.ndarray:
    if affine is None:
        return np.eye(4, dtype=np.float32)
    resolved = np.asarray(affine, dtype=np.float32)
    if resolved.shape != (4, 4):
        raise ConfigError("smooth_mask() affine must have shape (4, 4).")
    if not np.all(np.isfinite(resolved)):
        raise ConfigError("smooth_mask() affine values must be finite.")
    return resolved


def _take_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    slicer = [slice(None)] * np.asarray(array).ndim
    slicer[axis] = index
    return array[tuple(slicer)]


def _take_display_slice(
    array: np.ndarray,
    *,
    axis: int,
    index: int,
    crop_info: CropInfo | None,
) -> np.ndarray:
    if crop_info is None:
        return _take_slice(array, axis, index)
    slicer = [slice(None)] * np.asarray(array).ndim
    for dim, (start, stop) in enumerate(crop_info):
        slicer[dim] = start + index if dim == axis else slice(start, stop)
    return array[tuple(slicer)]


def _display_volume_shape(source: BBImage) -> tuple[int, ...]:
    if source.crop_info is None:
        return source.volume.data.shape
    return tuple(stop - start for start, stop in source.crop_info)


def _slice_display_shape(source: BBImage, *, axis: int) -> tuple[int, int]:
    shape = _display_volume_shape(source)
    if len(shape) < 3:
        raise ConfigError("slice rendering requires at least a 3D image.")
    axes = tuple(item for item in range(3) if item != axis)
    voxel_sizes = _voxel_sizes(source.volume.affine)
    in_plane_sizes = voxel_sizes[list(axes)]
    baseline = float(np.min(in_plane_sizes[in_plane_sizes > 0]))
    return tuple(
        max(1, int(round(shape[item] * source.scale * (voxel_sizes[item] / baseline))))
        for item in axes
    )


def _voxel_sizes(affine: np.ndarray) -> np.ndarray:
    values = np.sqrt(np.sum(np.asarray(affine, dtype=np.float64)[:3, :3] ** 2, axis=0))
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        return np.ones(3, dtype=np.float64)
    return values


def _validate_output_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ConfigError("slice rendering output shape must contain two dimensions.")
    output = tuple(int(value) for value in shape)
    if any(value <= 0 for value in output):
        raise ConfigError("slice rendering output shape values must be positive.")
    return output  # type: ignore[return-value]


def _unsharp_mask_radius(scale: float) -> float:
    resolved_scale = _validate_scale(scale)
    return max(0.5, min(6.0, resolved_scale * 0.75))


def _unsharp_mask_threshold(scale: float) -> int:
    resolved_scale = _validate_scale(scale)
    return max(1, int(round(resolved_scale)))


def _pil_from_slice(slice_rgba: np.ndarray):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ConfigError("Pillow is required to render slices.") from exc
    oriented = np.rot90(slice_rgba, 1)
    return Image.fromarray(oriented.astype(np.uint8), mode="RGBA")


def _multiply_alpha(slice_rgba: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0, 1))
    if alpha == 1.0:
        return slice_rgba
    output = slice_rgba.copy()
    output[..., 3] = np.clip(output[..., 3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return output


def _outline_slice(slice_rgba: np.ndarray, *, width: int):
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise ConfigError("scipy is required for outline rendering.") from exc
    active = slice_rgba[..., 3] > 0
    eroded = ndimage.binary_erosion(active, iterations=max(width, 1), border_value=0)
    outline = active & ~eroded
    output = slice_rgba.copy()
    output[..., 3] = np.where(outline, slice_rgba[..., 3], 0)
    return output


def _label_rgba(data: np.ndarray, LUT: str | Path, *, alpha: float) -> np.ndarray:
    labels = np.rint(data).astype(np.int64)
    table = load_lookup_table(LUT)
    rgba = np.zeros(labels.shape + (4,), dtype=np.uint8)
    for label in np.unique(labels[labels != 0]):
        entry = table.get(int(label))
        if entry is None:
            continue
        mask = labels == label
        rgba[mask, 0] = entry.rgba[0]
        rgba[mask, 1] = entry.rgba[1]
        rgba[mask, 2] = entry.rgba[2]
        rgba[mask, 3] = int(np.clip(alpha, 0, 1) * entry.rgba[3])
    return rgba


def _readonly_array(array: np.ndarray) -> np.ndarray:
    output = np.asarray(array)
    if output.flags.writeable:
        output = output.copy()
        output.setflags(write=False)
    return output


def _bbi_path(path: str | Path) -> Path:
    output = Path(path).expanduser()
    if output.suffix != ".bbi":
        output = output.with_suffix(".bbi")
    return output


def _validate_bbi_load_options(
    *,
    LUT: str | Path | None,
    colormap: Any | None,
    threshold: tuple[float, float] | None,
    indices: Sequence[int] | str | None,
    scale: float | None,
    interp: Any | None,
    sharpen: int | None,
    bias_correction: bool,
) -> None:
    has_load_options = any(
        value is not None for value in (LUT, colormap, threshold, indices, scale, interp, sharpen)
    )
    if bias_correction or has_load_options:
        raise ConfigError(
            "BBI archives already contain processed data. Load the .bbi first, "
            "then edit its display metadata fields directly if needed."
        )


def _validate_kind(value: object) -> ImageKind:
    if value in {"intensity", "delineation"}:
        return value  # type: ignore[return-value]
    raise ConfigError(f"Unsupported BBI image kind: {value}")


def _transform_from_metadata(metadata: dict[str, Any]) -> TransformRecord | None:
    if not metadata:
        return None
    template = metadata.get("template")
    forward_transforms = _transform_files_from_metadata(metadata.get("forward_transforms"))
    inverse_transforms = _transform_files_from_metadata(metadata.get("inverse_transforms"))
    if not forward_transforms:
        return None
    return TransformRecord(
        warp=metadata.get("warp"),
        template=Path(template) if template else None,
        forward_transforms=forward_transforms,
        inverse_transforms=inverse_transforms,
        applied=bool(metadata.get("applied", False)),
        template_shape=_optional_int_tuple(metadata.get("template_shape")),
        template_affine=_optional_array(metadata.get("template_affine")),
    )


def _optional_tuple(value: object) -> tuple | None:
    if value is None:
        return None
    if isinstance(value, list | tuple):
        return tuple(value)
    return (value,)


def _optional_int_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise ConfigError("TransformRecord template_shape must be a sequence of integers.")
    return tuple(int(item) for item in value)


def _optional_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _transform_metadata(transform: TransformRecord | None) -> dict[str, Any]:
    if transform is None:
        return {}
    return {
        "engine": "ANTs",
        "warp": transform.warp,
        "template": str(transform.template) if transform.template is not None else None,
        "applied": transform.applied,
        "template_shape": list(transform.template_shape) if transform.template_shape else None,
        "template_affine": (
            np.asarray(transform.template_affine, dtype=np.float64).tolist()
            if transform.template_affine is not None
            else None
        ),
        "forward_transforms": _transform_files_metadata(transform.forward_transforms),
        "inverse_transforms": _transform_files_metadata(transform.inverse_transforms),
    }


def _transform_files_metadata(files: tuple[AntsTransformFile, ...]) -> list[dict[str, str]]:
    return [
        {
            "name": file.name,
            "data": base64.b64encode(file.data).decode("ascii"),
        }
        for file in files
    ]


def _transform_files_from_metadata(value: object) -> tuple[AntsTransformFile, ...]:
    if not isinstance(value, list):
        return ()
    files: list[AntsTransformFile] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        encoded = item.get("data")
        if not isinstance(name, str) or not isinstance(encoded, str):
            continue
        files.append(AntsTransformFile(name=name, data=base64.b64decode(encoded.encode("ascii"))))
    return tuple(files)


def _default_interp_for_kind(kind: ImageKind) -> str:
    return "linear" if kind == "intensity" else "nearestNeighbor"


def _reslice_interp(
    kind: ImageKind,
    interp: Any | None,
) -> str:
    value = interp or _default_interp_for_kind(kind)
    return normalize_ants_interpolator(value)


def _validate_sharpen(sharpen: int) -> int:
    if isinstance(sharpen, bool):
        raise ConfigError("sharpen must be a non-negative integer.")
    try:
        value = int(sharpen)
    except (TypeError, ValueError) as exc:
        raise ConfigError("sharpen must be a non-negative integer.") from exc
    if value != sharpen or value < 0:
        raise ConfigError("sharpen must be a non-negative integer.")
    return value


def _validate_scale(scale: float) -> float:
    if isinstance(scale, bool):
        raise ConfigError("scale must be a real number greater than 0.")
    try:
        value = float(scale)
    except (TypeError, ValueError) as exc:
        raise ConfigError("scale must be a real number greater than 0.") from exc
    if not np.isfinite(value) or value <= 0:
        raise ConfigError("scale must be a real number greater than 0.")
    return value


def _normalize_interp_name(interp: Any) -> str:
    name = getattr(interp, "name", None)
    if isinstance(name, str) and name:
        value = name.strip().upper()
    elif isinstance(interp, str):
        value = interp.strip().upper()
    else:
        raise ConfigError(
            "interp must be one of: NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, LANCZOS."
        )
    aliases = {
        "LINEAR": "BILINEAR",
        "TRILINEAR": "BILINEAR",
        "CUBIC": "BICUBIC",
    }
    value = aliases.get(value, value)
    if value not in {"NEAREST", "BOX", "BILINEAR", "HAMMING", "BICUBIC", "LANCZOS"}:
        raise ConfigError(
            "interp must be one of: NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, LANCZOS."
        )
    return value


def _pil_interp_filter(interp: str):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ConfigError("Pillow is required for slice interpolation.") from exc
    return getattr(Image.Resampling, _normalize_interp_name(interp))


def _transform_status(transform: TransformRecord | None) -> str:
    if transform is None:
        return "none"
    return "applied" if transform.applied else "pending"


def _metadata_text(image: BBImage) -> str:
    colormap_name = getattr(image.colormap, "name", None) if image.colormap is not None else None
    mask_text = _matrix_summary(image.mask) if image.mask is not None else "none"
    transform = image.transform_info
    transform_lines = ["transform:"]
    if transform is None:
        transform_lines.append("  status: none")
    else:
        template_affine_text = (
            _matrix_summary(transform.template_affine)
            if transform.template_affine is not None
            else "none"
        )
        transform_lines.extend(
            [
                f"  status: {_transform_status(transform)}",
                f"  warp: {transform.warp}",
                f"  template: {transform.template}",
                f"  forward_transforms: {len(transform.forward_transforms)} file(s)",
                f"  inverse_transforms: {len(transform.inverse_transforms)} file(s)",
                f"  template_volume: {_volume_summary(transform.template_volume)}",
                f"  template_shape: {transform.template_shape}",
                f"  template_affine: {template_affine_text}",
            ]
        )
    lines = [
        "BBImage metadata",
        f"name: {image.name}",
        f"path: {image.path}",
        f"kind: {image.kind}",
        "volume:",
        f"  data: {_matrix_summary(image.volume.data)}",
        f"  affine: {_matrix_summary(image.volume.affine)}",
        f"  voxel_sizes: {_voxel_size_summary(image.volume.affine)}",
        "display:",
        f"  scale: {image.scale}",
        f"  interp: {image.interp}",
        f"  sharpen: {image.sharpen}",
        f"  colormap: {colormap_name}",
        f"  LUT: {image.LUT}",
        f"  threshold: {image.threshold}",
        f"indices: {image.indices}",
        f"crop_info: {image.crop_info}",
        f"mask: {mask_text}",
        *transform_lines,
    ]
    return "\n".join(lines)


def _matrix_summary(value: np.ndarray) -> str:
    array = np.asarray(value)
    return f"matrix shape={array.shape}, dtype={array.dtype}"


def _volume_summary(volume: Volume | None) -> str:
    if volume is None:
        return "none"
    return f"data {_matrix_summary(volume.data)}, affine {_matrix_summary(volume.affine)}"


def _voxel_size_summary(affine: np.ndarray) -> tuple[float, ...]:
    values = _voxel_sizes(affine)
    return tuple(float(np.round(value, 6)) for value in values)


def _has_pending_transform(transform: TransformRecord | None) -> bool:
    return transform is not None and not transform.applied


def _target_volume(target: str | Path | BBImage) -> Volume:
    if isinstance(target, BBImage):
        return target.volume
    return load_volume(ensure_path(target))


def _target_template(target: str | Path | BBImage) -> Path | None:
    if isinstance(target, BBImage):
        return target.path if target.path.exists() else None
    return ensure_path(target)


def _template_volume_from_record(transform: TransformRecord) -> Volume:
    if transform.template_volume is not None:
        return transform.template_volume
    if transform.template_shape is not None and transform.template_affine is not None:
        return volume_from_data(
            transform.template or Path("template_space.nii.gz"),
            np.zeros(transform.template_shape, dtype=np.float32),
            transform.template_affine,
        )
    if transform.template is not None and transform.template.exists():
        return load_volume(transform.template)
    raise ConfigError(
        "TransformRecord requires template shape/affine metadata, an in-memory template volume, "
        "or an existing template path."
    )


def _pending_transform_record(
    transform: TransformRecord,
) -> TransformRecord:
    _validate_transform_files(transform)
    return TransformRecord(
        warp=transform.warp,
        template=transform.template,
        forward_transforms=transform.forward_transforms,
        inverse_transforms=transform.inverse_transforms,
        applied=False,
        template_volume=_template_volume_from_record(transform),
    )


def _applied_transform_record(
    transform: TransformRecord,
) -> TransformRecord:
    return TransformRecord(
        warp=transform.warp,
        template=transform.template,
        forward_transforms=transform.forward_transforms,
        inverse_transforms=transform.inverse_transforms,
        applied=True,
        template_volume=_template_volume_from_record(transform),
    )


def _validate_transform_files(transform: TransformRecord) -> None:
    if not transform.forward_transforms:
        raise ConfigError("TransformRecord does not contain ANTs transform files.")


def _image_with_pending_transform(
    source: BBImage,
    *,
    transform_info: TransformRecord,
) -> BBImage:
    _validate_transform_files(transform_info)
    return source._with_data(
        data=source.volume.data,
        volume=source.volume,
        transform_info=transform_info,
        crop_info=None,
        mask="keep",
    )


def _reslice_image(
    source: BBImage,
    *,
    interp: Any | None,
    cache_dir: str | Path | None,
    debug: bool,
) -> BBImage:
    if not isinstance(source, BBImage):
        raise ConfigError("reslice() source must be a BBImage.")
    if source.transform_info is None:
        if interp is not None:
            normalize_ants_interpolator(interp)
        return _copy_image(source)
    if source.transform_info.applied:
        if interp is not None:
            raise ConfigError("reslice() cannot change interpolation after a transform is applied.")
        return _copy_image(source)
    transform_info = source.transform_info
    _validate_transform_files(transform_info)
    fixed_volume = _template_volume_from_record(transform_info)
    default_interp = source.interp if source.kind == "intensity" else None
    resolved_interp = _reslice_interp(source.kind, interp if interp is not None else default_interp)
    transformed_image = apply_ants_transform(
        fixed=fixed_volume,
        moving=source.volume,
        transforms=transform_info.forward_transforms,
        interp=resolved_interp,
        cache_dir=cache_dir,
        debug=debug,
    )
    transformed_mask = _reslice_mask(
        source,
        fixed_volume=fixed_volume,
        transform_info=transform_info,
        interp=resolved_interp,
        cache_dir=cache_dir,
        debug=debug,
    )
    return _image_from_resliced_result(
        source,
        image=transformed_image,
        transform_info=_applied_transform_record(transform_info),
        mask=transformed_mask,
    )


def _image_from_resliced_result(
    source: BBImage,
    *,
    image,
    transform_info: TransformRecord,
    mask: np.ndarray | None,
) -> BBImage:
    transformed_volume = volume_from_image(source.path, image)
    transformed_data = _coerce_transformed_data(
        transformed_volume.data,
        kind=source.kind,
    )
    resliced = source._with_data(
        path=source.path,
        volume=Volume(
            path=source.path,
            image=transformed_volume.image,
            data=transformed_data,
            affine=transformed_volume.affine,
        ),
        data=transformed_data,
        transform_info=transform_info,
        crop_info=source.crop_info,
        mask=_normalize_mask(mask, shape=transformed_data.shape),
    )
    return resliced


def _copy_image(source: BBImage) -> BBImage:
    return BBImage(
        path=source.path,
        volume=source.volume,
        LUT=source.LUT,
        colormap=source.colormap,
        indices=source.indices,
        threshold=source.threshold,
        crop_info=source.crop_info,
        mask=source.mask,
        transform_info=source.transform_info,
        name=source.name,
        kind=source.kind,
        scale=source.scale,
        interp=source.interp,
        sharpen=source.sharpen,
    )


def _import_scipy_ndimage():
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise ConfigError("scipy is required for mask enhancement.") from exc
    return ndimage


def _reslice_mask(
    source: BBImage,
    *,
    fixed_volume: Volume,
    transform_info: TransformRecord,
    interp: str,
    cache_dir: str | Path | None,
    debug: bool,
) -> np.ndarray | None:
    if source.mask is None:
        return None
    mask_volume = Volume(
        path=source.path,
        image=source.volume.image,
        data=source.mask.astype(np.float32),
        affine=source.volume.affine,
    )
    transformed_image = apply_ants_transform(
        fixed=fixed_volume,
        moving=mask_volume,
        transforms=transform_info.forward_transforms,
        interp=interp,
        cache_dir=cache_dir,
        debug=debug,
    )
    transformed_volume = volume_from_image(source.path, transformed_image)
    mask = np.nan_to_num(transformed_volume.data, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(mask, 0, 1).astype(np.float16, copy=True)


def _as_transform_record(matrix: TransformRecord | BBImage) -> TransformRecord:
    if isinstance(matrix, TransformRecord):
        return matrix
    if isinstance(matrix, BBImage) and matrix.transform_info is not None:
        return matrix.transform_info
    raise ConfigError("matrix must be a TransformRecord or a BBImage with transform_info.")
