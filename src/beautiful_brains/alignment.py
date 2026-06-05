"""ANTsPy-backed spatial registration and bias-field correction."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from beautiful_brains.exceptions import ConfigError, DependencyError
from beautiful_brains.io import Volume, image_from_volume_data, volume_from_image

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(os.cpu_count() or 1))

_ALLOWED_WARPS: tuple[str, ...] = ("Rigid", "Affine", "QuickRigid")
_REQUIRED_ANTS_ATTRIBUTES: tuple[str, ...] = (
    "registration",
    "apply_transforms",
    "image_read",
    "image_write",
    "get_mask",
    "n4_bias_field_correction",
    "crop_indices",
    "threshold_image",
    "iMath",
)
_ANTS_TOOLS: AntsTools | None = None
CropInfo = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class AntsTransformFile:
    """A reusable ANTs transform file captured from ANTsPy output."""

    name: str
    data: bytes


@dataclass(frozen=True)
class AntsRegistrationResult:
    """ANTs registration transform files needed by Beautiful-Brains."""

    forward_transforms: tuple[AntsTransformFile, ...]
    inverse_transforms: tuple[AntsTransformFile, ...]


@dataclass(frozen=True)
class AntsCropResult:
    """ANTs crop output and the index bounds used to create it."""

    volume: Volume
    crop_info: CropInfo


@dataclass(frozen=True)
class AntsTools:
    """The small ANTsPy API surface used by Beautiful-Brains."""

    registration: Callable[..., Any]
    apply_transforms: Callable[..., Any]
    image_read: Callable[..., Any]
    image_write: Callable[..., Any]
    get_mask: Callable[..., Any]
    n4_bias_field_correction: Callable[..., Any]
    crop_indices: Callable[..., Any]
    threshold_image: Callable[..., Any]
    iMath: Callable[..., Any]
    get_ants_data: Callable[..., Any] | None = None
    from_numpy_like: Callable[..., Any] | None = None
    from_nibabel: Callable[..., Any] | None = None
    to_nibabel: Callable[..., Any] | None = None


def register_ants_transform(
    *,
    fixed: Volume,
    moving: Volume,
    warp: str,
    cache_dir: str | Path | None = None,
    debug: bool = False,
) -> AntsRegistrationResult:
    """Estimate an ANTs transform without resampling the moving image."""

    ants = _import_ants()
    resolved_warp = normalize_ants_warp(warp)
    with _temporary_work_dir(cache_dir) as work_dir:
        fixed_ants = _ants_from_volume(ants, fixed, work_dir, "fixed")
        moving_ants = _ants_from_volume(ants, moving, work_dir, "moving")
        _debug_print(debug, f"estimating ANTs {resolved_warp} transform")
        result = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform=resolved_warp,
            outprefix=str(work_dir / "reg_"),
            verbose=debug,
        )
        forward_paths = _transform_paths(result.get("fwdtransforms"))
        inverse_paths = _transform_paths(result.get("invtransforms"))
        return AntsRegistrationResult(
            forward_transforms=_capture_transform_files(forward_paths),
            inverse_transforms=_capture_transform_files(inverse_paths),
        )


def apply_ants_transform(
    *,
    fixed: Volume,
    moving: Volume,
    transforms: tuple[AntsTransformFile, ...],
    interp: str,
    cache_dir: str | Path | None = None,
    debug: bool = False,
):
    """Apply captured ANTs transforms to a moving image."""

    if not transforms:
        raise ConfigError("TransformRecord does not contain ANTs transform files.")
    ants = _import_ants()
    resolved_interp = normalize_ants_interpolator(interp)
    with _temporary_work_dir(cache_dir) as work_dir:
        fixed_ants = _ants_from_volume(ants, fixed, work_dir, "fixed")
        moving_ants = _ants_from_volume(ants, moving, work_dir, "moving")
        transform_paths = _write_transform_files(transforms, work_dir)
        _debug_print(debug, "applying ANTs transform")
        transformed_ants = ants.apply_transforms(
            fixed=fixed_ants,
            moving=moving_ants,
            transformlist=[str(path) for path in transform_paths],
            interpolator=resolved_interp,
            verbose=debug,
        )
        return _ants_to_nibabel(ants, transformed_ants, work_dir, "transformed")


def bias_correct_volume(
    volume: Volume,
    *,
    cache_dir: str | Path | None = None,
    debug: bool = False,
) -> Volume:
    """Apply ANTs N4 bias-field correction to a scalar image volume."""

    ants = _import_ants()
    with _temporary_work_dir(cache_dir) as work_dir:
        image = _ants_from_volume(ants, volume, work_dir, "bias_input")
        mask = ants.get_mask(image)
        _debug_print(debug, "applying ANTs N4 bias-field correction")
        corrected = ants.n4_bias_field_correction(
            image,
            mask=mask,
            shrink_factor=4,
            convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
            spline_param=200,
            return_bias_field=False,
            verbose=debug,
        )
        corrected_image = _ants_to_nibabel(ants, corrected, work_dir, "bias_corrected")
    return volume_from_image(volume.path, corrected_image)


def create_mask_array(
    volume: Volume,
    *,
    pad: int = 0,
    low_thresh: float | None = None,
    max_thresh: float | None = None,
    cleanup: int | None = None,
    cache_dir: str | Path | None = None,
    debug: bool = False,
) -> np.ndarray:
    """Create a float16 brain mask from a volume using ANTs."""

    resolved_pad = _validate_mask_pad(pad)
    get_mask_kwargs = _mask_kwargs(
        low_thresh=low_thresh,
        max_thresh=max_thresh,
        cleanup=cleanup,
    )
    ants = _import_ants()
    with _temporary_work_dir(cache_dir) as work_dir:
        image = _ants_from_volume(ants, volume, work_dir, "mask_input")
        _debug_print(debug, f"creating ANTs mask with pad={resolved_pad}")
        mask = ants.get_mask(image, **get_mask_kwargs)
        if resolved_pad:
            mask = ants.iMath(mask, "MD", resolved_pad)
    return np.asarray(mask.numpy(), dtype=np.float16)


def create_brainmask_array(
    volume: Volume,
    *,
    modality: str = "t1",
    threshold: float = 0.5,
    pad: int = 0,
    cleanup: int = 0,
    cache_dir: str | Path | None = None,
    debug: bool = False,
) -> np.ndarray:
    """Create a float16 brain-only mask from a volume using ANTsPyNet."""

    resolved_modality = _validate_brainmask_modality(modality)
    resolved_threshold = _validate_brainmask_threshold(threshold)
    resolved_pad = _validate_brainmask_pad(pad)
    resolved_cleanup = _validate_brainmask_cleanup(cleanup)
    brain_extraction = _import_antspynet_brain_extraction()
    ants = _import_ants()
    with _temporary_work_dir(cache_dir) as work_dir:
        image = _ants_from_volume(ants, volume, work_dir, "brainmask_input")
        _debug_print(debug, f"creating ANTsPyNet brain mask with modality={resolved_modality!r}")
        probability = brain_extraction(image, modality=resolved_modality, verbose=debug)
        mask = ants.threshold_image(probability, resolved_threshold, 1.0)
        if resolved_cleanup:
            mask = ants.iMath(mask, "ME", resolved_cleanup)
            mask = ants.iMath(mask, "GetLargestComponent")
            mask = ants.iMath(mask, "MD", resolved_cleanup)
            mask = ants.iMath(mask, "FillHoles")
        if resolved_pad:
            mask = ants.iMath(mask, "MD", resolved_pad)
    return np.asarray(np.asarray(mask.numpy()) > 0, dtype=np.float16)


def crop_volume(
    volume: Volume,
    *,
    how: str = "mean",
    pad: int = 0,
    crop_info: CropInfo | None = None,
    cache_dir: str | Path | None = None,
    debug: bool = False,
) -> AntsCropResult:
    """Crop a volume with ANTs and return the crop bounds that were used."""

    resolved_crop_info = validate_crop_info(crop_info, shape=volume.data.shape)
    resolved_how = normalize_crop_method(how) if resolved_crop_info is None else None
    resolved_pad = _validate_crop_pad(pad) if resolved_crop_info is None else 0
    ants = _import_ants()
    with _temporary_work_dir(cache_dir) as work_dir:
        image = _ants_from_volume(ants, volume, work_dir, "crop_input")
        if resolved_crop_info is None:
            _debug_print(debug, f"building {resolved_how} crop mask with pad={resolved_pad}")
            mask = _crop_mask(ants, image, how=resolved_how or "mean")
            if resolved_pad:
                mask = ants.iMath(mask, "MD", resolved_pad)
            resolved_crop_info = _crop_info_from_mask(mask)
        else:
            _debug_print(debug, f"using provided crop_info={resolved_crop_info}")
        lower, upper = _crop_bounds(resolved_crop_info)
        cropped = ants.crop_indices(image, lower, upper)
        cropped_image = _ants_to_nibabel(ants, cropped, work_dir, "cropped")
    return AntsCropResult(
        volume=volume_from_image(volume.path, cropped_image),
        crop_info=resolved_crop_info,
    )


def normalize_ants_warp(warp: str | None) -> str:
    if warp is None:
        raise ConfigError("transform() requires warp='Rigid', warp='Affine', or warp='QuickRigid'.")
    if warp not in _ALLOWED_WARPS:
        raise ConfigError("transform() warp must be one of: Rigid, Affine, QuickRigid.")
    return warp


def normalize_crop_method(how: str) -> str:
    normalized = how.strip().lower()
    if normalized in {"mean", "otsu"}:
        return normalized
    raise ConfigError("crop() how must be 'mean' or 'otsu'.")


def normalize_ants_interpolator(interp: Any) -> str:
    normalized = _interpolator_name(interp).lower()
    aliases = {
        "linear": "linear",
        "trilinear": "linear",
        "gaussian": "gaussian",
        "nearest": "nearestNeighbor",
        "nearestneighbor": "nearestNeighbor",
        "nearestneighbour": "nearestNeighbor",
        "nn": "nearestNeighbor",
        "multilabel": "multiLabel",
        "multi_label": "multiLabel",
        "label": "multiLabel",
        "genericlabel": "genericLabel",
        "generic_label": "genericLabel",
        "bspline": "bSpline",
        "b-spline": "bSpline",
        "spline": "bSpline",
        "bicubic": "bSpline",
        "cubic": "bSpline",
        "lanczos": "lanczosWindowedSinc",
        "lanczoswindowedsinc": "lanczosWindowedSinc",
        "lanczos_windowed_sinc": "lanczosWindowedSinc",
        "cosinewindowedsinc": "cosineWindowedSinc",
        "cosine_windowed_sinc": "cosineWindowedSinc",
        "welchwindowedsinc": "welchWindowedSinc",
        "welch_windowed_sinc": "welchWindowedSinc",
        "hammingwindowedsinc": "hammingWindowedSinc",
        "hamming_windowed_sinc": "hammingWindowedSinc",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ConfigError(
            "ANTs interpolation must be one of: linear, nearestNeighbor, multiLabel, "
            "genericLabel, bSpline, gaussian, cosineWindowedSinc, welchWindowedSinc, "
            "hammingWindowedSinc, lanczosWindowedSinc, or Pillow Resampling.BICUBIC/LANCZOS."
        ) from exc


def _interpolator_name(interp: Any) -> str:
    name = getattr(interp, "name", None)
    if isinstance(name, str) and name:
        return name.strip()
    if isinstance(interp, str):
        return interp.strip()
    raise ConfigError("ANTs interpolation must be a string or Pillow Image.Resampling value.")


def _import_ants() -> AntsTools:
    global _ANTS_TOOLS
    if _ANTS_TOOLS is not None:
        return _ANTS_TOOLS
    _clear_incomplete_ants_import()
    try:
        from ants.core.ants_image_io import from_numpy_like, image_read, image_write
        from ants.ops.bias_correction import n4_bias_field_correction
        from ants.ops.crop_image import crop_indices
        from ants.ops.get_mask import get_mask
        from ants.ops.iMath import iMath
        from ants.ops.threshold_image import threshold_image
        from ants.registration.apply_transforms import apply_transforms
        from ants.registration.registration import registration
        from ants.utils.get_ants_data import get_ants_data
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "ANTsPy is required for registration and bias correction. Install antspyx."
        ) from exc
    ants_module = sys.modules.get("ants")
    _install_ants_compatibility(ants_module, from_numpy_like=from_numpy_like)
    _ANTS_TOOLS = AntsTools(
        registration=registration,
        apply_transforms=apply_transforms,
        image_read=image_read,
        image_write=image_write,
        get_mask=get_mask,
        n4_bias_field_correction=n4_bias_field_correction,
        crop_indices=crop_indices,
        threshold_image=threshold_image,
        iMath=iMath,
        get_ants_data=get_ants_data,
        from_numpy_like=from_numpy_like,
        from_nibabel=getattr(ants_module, "from_nibabel", None),
        to_nibabel=getattr(ants_module, "to_nibabel", None),
    )
    return _ANTS_TOOLS


def _clear_incomplete_ants_import() -> None:
    module = sys.modules.get("ants")
    if module is None:
        return
    if all(hasattr(module, name) for name in _REQUIRED_ANTS_ATTRIBUTES):
        return
    for name in tuple(sys.modules):
        if name == "ants" or name.startswith("ants."):
            del sys.modules[name]


def _install_ants_compatibility(ants_module, *, from_numpy_like: Callable[..., Any]) -> None:
    if ants_module is None:
        return
    if not hasattr(ants_module, "from_numpy_like"):
        ants_module.from_numpy_like = from_numpy_like
    if not hasattr(ants_module, "segmentation_to_one_hot"):
        ants_module.segmentation_to_one_hot = _segmentation_to_one_hot
    if not hasattr(ants_module, "one_hot_to_segmentation"):

        def one_hot_to_segmentation(
            one_hot_array,
            domain_image,
            channel_first_ordering: bool = False,
        ):
            return _one_hot_to_segmentation(
                one_hot_array,
                domain_image,
                channel_first_ordering=channel_first_ordering,
                from_numpy_like=from_numpy_like,
            )

        ants_module.one_hot_to_segmentation = one_hot_to_segmentation


def _segmentation_to_one_hot(
    segmentations_array,
    segmentation_labels=None,
    channel_first_ordering: bool = False,
):
    labels = (
        np.unique(segmentations_array)
        if segmentation_labels is None
        else tuple(segmentation_labels)
    )
    if len(labels) < 2:
        raise ValueError("At least two segmentation labels need to be specified.")
    one_hot = np.stack(
        [(np.asarray(segmentations_array) == label).astype(np.float32) for label in labels],
        axis=-1,
    )
    if channel_first_ordering:
        if one_hot.ndim == 3:
            one_hot = one_hot.transpose((2, 0, 1))
        elif one_hot.ndim == 4:
            one_hot = one_hot.transpose((3, 0, 1, 2))
        else:
            raise ValueError("Unrecognized image dimensionality.")
    return one_hot


def _one_hot_to_segmentation(
    one_hot_array,
    domain_image,
    *,
    channel_first_ordering: bool = False,
    from_numpy_like: Callable[..., Any],
):
    array = np.asarray(one_hot_array)
    if channel_first_ordering:
        number_of_labels = array.shape[0]
    else:
        number_of_labels = array.shape[-1]

    probability_images = []
    for label in range(number_of_labels):
        if channel_first_ordering:
            image_array = np.squeeze(array[label, ...])
        else:
            image_array = np.squeeze(array[..., label])
        probability_images.append(
            from_numpy_like(np.asarray(image_array, dtype=np.float32), domain_image)
        )
    return probability_images


def _crop_mask(ants: AntsTools, image, *, how: str):
    if how == "mean":
        return ants.get_mask(image)
    otsu_labels = ants.threshold_image(image, "Otsu", 3)
    return ants.threshold_image(otsu_labels, 1, 3)


def normalize_crop_info(crop_info: object) -> CropInfo | None:
    """Normalize crop bounds to exclusive index pairs."""

    if crop_info is None:
        return None
    try:
        items = tuple(crop_info)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ConfigError(
            "crop_info must contain one [start, stop] pair per image dimension."
        ) from exc
    bounds: list[tuple[int, int]] = []
    for item in items:
        try:
            pair = tuple(item)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ConfigError("Each crop_info entry must be a [start, stop] pair.") from exc
        if len(pair) != 2:
            raise ConfigError("Each crop_info entry must be a [start, stop] pair.")
        start = _crop_index(pair[0])
        stop = _crop_index(pair[1])
        if start < 0 or stop <= start:
            raise ConfigError("crop_info bounds must satisfy 0 <= start < stop.")
        bounds.append((start, stop))
    return tuple(bounds)


def validate_crop_info(crop_info: object, *, shape: tuple[int, ...]) -> CropInfo | None:
    """Normalize crop bounds and validate them against an image shape."""

    bounds = normalize_crop_info(crop_info)
    if bounds is None:
        return None
    if len(bounds) != len(shape):
        raise ConfigError("crop_info must contain one [start, stop] pair per image dimension.")
    for axis, (_, stop) in enumerate(bounds):
        if stop > shape[axis]:
            raise ConfigError(
                "crop_info bounds must satisfy 0 <= start < stop <= image dimension size."
            )
    return bounds


def _crop_info_from_mask(mask) -> CropInfo:
    active = np.argwhere(np.asarray(mask.numpy()) > 0)
    if active.size == 0:
        raise ConfigError("crop() mask is empty; cannot determine crop bounds.")
    lower = active.min(axis=0)
    upper = active.max(axis=0) + 1
    return tuple((int(start), int(stop)) for start, stop in zip(lower, upper, strict=True))


def _crop_bounds(crop_info: CropInfo) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lower = tuple(start for start, _ in crop_info)
    upper = tuple(stop for _, stop in crop_info)
    return lower, upper


def _crop_index(value: object) -> int:
    if isinstance(value, bool):
        raise ConfigError("crop_info indices must be integers.")
    try:
        index = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError("crop_info indices must be integers.") from exc
    if index != value:
        raise ConfigError("crop_info indices must be integers.")
    return index


def _validate_crop_pad(pad: int) -> int:
    return _validate_non_negative_integer(pad, name="crop() pad")


def _validate_mask_pad(pad: int) -> int:
    return _validate_non_negative_integer(pad, name="create_mask() pad")


def _validate_brainmask_pad(pad: int) -> int:
    return _validate_non_negative_integer(pad, name="create_brainmask() pad")


def _validate_mask_cleanup(cleanup: int | None) -> int | None:
    if cleanup is None:
        return None
    return _validate_non_negative_integer(cleanup, name="create_mask() cleanup")


def _validate_brainmask_cleanup(cleanup: int) -> int:
    return _validate_non_negative_integer(cleanup, name="create_brainmask() cleanup")


def _validate_mask_low_thresh(low_thresh: float | None) -> float | None:
    return _validate_optional_finite_number(
        low_thresh,
        name="create_mask() low_thresh",
    )


def _validate_mask_max_thresh(max_thresh: float | None) -> float | None:
    return _validate_optional_finite_number(
        max_thresh,
        name="create_mask() max_thresh",
    )


def _validate_optional_finite_number(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a finite number.") from exc
    if not np.isfinite(number):
        raise ConfigError(f"{name} must be a finite number.")
    return number


def _mask_kwargs(
    *,
    low_thresh: float | None,
    max_thresh: float | None,
    cleanup: int | None,
) -> dict[str, float | int]:
    kwargs: dict[str, float | int] = {}
    resolved_low_thresh = _validate_mask_low_thresh(low_thresh)
    resolved_max_thresh = _validate_mask_max_thresh(max_thresh)
    resolved_cleanup = _validate_mask_cleanup(cleanup)
    if resolved_low_thresh is not None:
        kwargs["low_thresh"] = resolved_low_thresh
    if resolved_max_thresh is not None:
        kwargs["high_thresh"] = resolved_max_thresh
    if resolved_cleanup is not None:
        kwargs["cleanup"] = resolved_cleanup
    return kwargs


def _validate_brainmask_modality(modality: str) -> str:
    if not isinstance(modality, str) or not modality.strip():
        raise ConfigError("create_brainmask() modality must be a non-empty string.")
    return modality.strip()


def _validate_brainmask_threshold(threshold: float) -> float:
    number = _validate_optional_finite_number(
        threshold,
        name="create_brainmask() threshold",
    )
    if number is None or number <= 0 or number >= 1:
        raise ConfigError("create_brainmask() threshold must be between 0 and 1.")
    return number


def _validate_non_negative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a non-negative integer.")
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a non-negative integer.") from exc
    if integer != value or integer < 0:
        raise ConfigError(f"{name} must be a non-negative integer.")
    return integer


@contextmanager
def _temporary_work_dir(cache_dir: str | Path | None = None) -> Iterator[Path]:
    base = Path(cache_dir).expanduser() if cache_dir is not None else _default_temp_dir()
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="beautiful_brains_ants_", dir=base) as path:
        yield Path(path)


def _ants_from_volume(ants, volume: Volume, work_dir: Path, name: str):
    image = image_from_volume_data(volume)
    if ants.from_nibabel is not None:
        return ants.from_nibabel(image)
    nib = _import_nibabel()
    path = work_dir / f"{name}.nii.gz"
    nib.save(image, str(path))
    return ants.image_read(str(path))


def _ants_to_nibabel(ants, image, work_dir: Path, name: str):
    if ants.to_nibabel is not None:
        return ants.to_nibabel(image)
    nib = _import_nibabel()
    path = work_dir / f"{name}.nii.gz"
    ants.image_write(image, str(path))
    loaded = nib.load(str(path))
    header = getattr(loaded, "header", None)
    header = header.copy() if header is not None else None
    return nib.Nifti1Image(loaded.get_fdata(dtype="float32"), loaded.affine, header=header)


def _import_antspynet_brain_extraction():
    try:
        from antspynet import brain_extraction
    except ModuleNotFoundError:
        try:
            from antspynet.utilities import brain_extraction
        except ModuleNotFoundError as exc:
            raise DependencyError(
                "create_brainmask() requires ANTsPyNet. Install it with "
                "`python -m pip install antspynet`."
            ) from exc
    return brain_extraction


def _import_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "nibabel is required to convert images between Beautiful-Brains and ANTs."
        ) from exc
    return nib


def _transform_paths(paths: object) -> tuple[Path, ...]:
    if paths is None:
        return ()
    if isinstance(paths, str):
        values = (paths,)
    else:
        values = tuple(paths)
    return tuple(Path(value) for value in values)


def _capture_transform_files(paths: tuple[Path, ...]) -> tuple[AntsTransformFile, ...]:
    files: list[AntsTransformFile] = []
    for path in paths:
        if not path.exists():
            raise DependencyError(f"ANTs did not produce transform file: {path}")
        files.append(AntsTransformFile(name=path.name, data=path.read_bytes()))
    return tuple(files)


def _write_transform_files(
    transforms: tuple[AntsTransformFile, ...],
    work_dir: Path,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, transform in enumerate(transforms):
        path = work_dir / f"{index}_{Path(transform.name).name}"
        path.write_bytes(transform.data)
        paths.append(path)
    return tuple(paths)


def _default_temp_dir() -> Path:
    return Path(os.environ.get("TMPDIR") or tempfile.gettempdir()).expanduser()


def _debug_print(debug: bool, message: str) -> None:
    if debug:
        print(f"[beautiful-brains] {message}")
