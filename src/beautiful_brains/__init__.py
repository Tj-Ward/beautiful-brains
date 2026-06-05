"""Beautiful-Brains public API.

Notebook-first usage:

    import beautiful_brains as bb
    import matplotlib as mpl

    mri = bb.load("nu.nii.gz", colormap=mpl.colormaps["gray"], threshold=(10, 200))
    fig = bb.bbfigure(grid=(1, 1), dpi=300)
    fig[0, 0] = mri.slice(80, plane="axial")
    image = fig.render()
"""

from importlib.metadata import PackageNotFoundError, version

from beautiful_brains.colormaps import colorbar
from beautiful_brains.figure import BBFigure, BBPanel, bbfigure, make_video
from beautiful_brains.image import (
    BBImage,
    TransformRecord,
    apply_transform,
    create_brainmask,
    create_mask,
    crop,
    load,
    smooth_image,
    smooth_mask,
    transform,
)

try:
    __version__ = version("beautiful-brains")
except PackageNotFoundError:
    __version__ = "0.0.1"

__all__ = [
    "__version__",
    "BBFigure",
    "BBImage",
    "BBPanel",
    "TransformRecord",
    "apply_transform",
    "bbfigure",
    "colorbar",
    "create_brainmask",
    "create_mask",
    "crop",
    "load",
    "make_video",
    "smooth_image",
    "smooth_mask",
    "transform",
]
