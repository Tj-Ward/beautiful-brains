"""NIfTI loading and smoothing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beautiful_brains.exceptions import DependencyError, ImageGeometryError
from beautiful_brains.utils import resolve_existing_nifti


@dataclass(frozen=True)
class Volume:
    path: Path
    image: object
    data: np.ndarray
    affine: np.ndarray


def _import_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "nibabel is required to read NIfTI images. Install Beautiful-Brains dependencies."
        ) from exc
    return nib


def _import_scipy_ndimage():
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "scipy is required for smoothing and mask outlines. "
            "Install Beautiful-Brains dependencies."
        ) from exc
    return ndimage


def load_volume(path: Path, *, canonical: bool = True) -> Volume:
    path = resolve_existing_nifti(path)
    nib = _import_nibabel()
    image = nib.load(str(path))
    if canonical:
        image = nib.as_closest_canonical(image)
    data = np.asarray(image.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ImageGeometryError(f"Only 3D images are supported right now: {path}")
    return Volume(path=path, image=image, data=data, affine=np.asarray(image.affine))


def image_from_volume_data(volume: Volume, data: np.ndarray | None = None):
    """Create a nibabel image using the affine/header geometry from a volume."""

    nib = _import_nibabel()
    header = getattr(volume.image, "header", None)
    header = header.copy() if header is not None else None
    array = volume.data if data is None else data
    return nib.Nifti1Image(np.asarray(array, dtype=np.float32), volume.affine, header=header)


def volume_from_image(path: Path, image) -> Volume:
    data = np.asarray(image.get_fdata(dtype=np.float32))
    return Volume(path=path, image=image, data=data, affine=np.asarray(image.affine))


def volume_from_data(path: Path, data: np.ndarray, affine: np.ndarray) -> Volume:
    nib = _import_nibabel()
    data = np.asarray(data, dtype=np.float32)
    affine = np.asarray(affine, dtype=np.float64)
    image = nib.Nifti1Image(data, affine)
    return Volume(path=path, image=image, data=data, affine=affine)


def voxel_sizes(affine: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))


def smooth_fwhm(
    data: np.ndarray,
    affine: np.ndarray,
    fwhm_mm: float | Sequence[float] | None,
) -> np.ndarray:
    if fwhm_mm is None:
        return data
    ndimage = _import_scipy_ndimage()
    fwhm_over_sigma = np.sqrt(8 * np.log(2))
    fwhm = np.asarray(fwhm_mm, dtype=np.float32)
    if fwhm.ndim == 0:
        fwhm = np.repeat(fwhm, 3)
    if fwhm.shape != (3,):
        raise ValueError("Smoothing FWHM must be a scalar or a 3-value sequence.")
    if np.any(fwhm < 0):
        raise ValueError("Smoothing FWHM must be non-negative.")
    if np.all(fwhm <= 0):
        return data
    sigma = fwhm / (fwhm_over_sigma * voxel_sizes(affine))
    finite = np.isfinite(data)
    filled = np.where(finite, data, 0)
    weights = ndimage.gaussian_filter(finite.astype(np.float32), sigma=sigma, mode="nearest")
    smoothed = ndimage.gaussian_filter(filled, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed / weights
    smoothed[weights == 0] = np.nan
    return smoothed.astype(np.float32)
