from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

import beautiful_brains as bb
import beautiful_brains.alignment as alignment_module
import beautiful_brains.colormaps as colormap_module
import beautiful_brains.figure as figure_module
import beautiful_brains.image as image_module
from beautiful_brains.alignment import AntsCropResult, AntsRegistrationResult, AntsTransformFile
from beautiful_brains.exceptions import ConfigError, DependencyError
from beautiful_brains.io import image_from_volume_data, volume_from_data


def _write_nifti(path, data, affine=None):
    nib.save(
        nib.Nifti1Image(
            np.asarray(data, dtype=np.float32),
            np.eye(4) if affine is None else np.asarray(affine, dtype=np.float32),
        ),
        path,
    )
    return path


def _intensity_image(tmp_path):
    data = np.zeros((8, 9, 10), dtype=np.float32)
    data[2:7, 3:8, 4:9] = 100
    return bb.load(_write_nifti(tmp_path / "mri.nii.gz", data))


def _delineation_image(tmp_path):
    data = np.zeros((8, 9, 10), dtype=np.float32)
    data[2:7, 3:8, 4:9] = 1
    data[3, 4, 5] = 2
    return bb.load(_write_nifti(tmp_path / "seg.nii.gz", data), LUT="freesurfer", indices=(1,))


def _animation_frames():
    from PIL import Image

    return [
        Image.new("RGBA", (5, 6), (255, 0, 0, 255)),
        Image.new("RGBA", (5, 6), (0, 255, 0, 192)),
        Image.new("RGBA", (5, 6), (0, 0, 255, 128)),
    ]


def _animation_figures():
    figures = []
    for image in _animation_frames():
        figure = bb.bbfigure(size=np.sqrt(2) / 10, grid=(1, 1), dpi=100, background=(0, 0, 0, 0))
        figure[0] = image
        figures.append(figure)
    return figures


def test_load_intensity_and_slice(tmp_path):
    image = _intensity_image(tmp_path)

    assert image.kind == "intensity"
    assert image.LUT is None
    assert image.threshold is not None
    assert not hasattr(image, "rgba")
    assert image.scale == 3.0
    assert image.interp == "BICUBIC"
    assert image.sharpen == 0

    axial = image.slice(5, plane="axial")
    assert axial.mode == "RGBA"
    assert axial.size == (image.volume.data.shape[0] * 3, image.volume.data.shape[1] * 3)

    loaded = bb.load(_write_nifti(tmp_path / "scaled.nii.gz", image.volume.data), scale=2)
    assert loaded.slice(5, plane="axial").size == (16, 18)


def test_slice_respects_affine_spacing_and_resamples_mask_for_display(tmp_path):
    data = np.ones((8, 9, 10), dtype=np.float32)
    affine = np.diag([2, 1, 1, 1]).astype(np.float32)
    image = bb.load(_write_nifti(tmp_path / "anisotropic.nii.gz", data, affine), scale=2)
    mask = np.ones(image.volume.data.shape, dtype=np.float16)
    mask[:4, :, :] = np.float16(0.25)

    image.mask = mask
    rendered = image.slice(5, plane="axial")
    alpha = np.asarray(rendered)[..., 3]

    assert image.mask.shape == image.volume.data.shape
    assert rendered.size == (32, 18)
    assert alpha.min() < 128
    assert alpha.max() == 255


def test_load_delineation_uses_lut_and_indices(tmp_path):
    data = np.zeros((8, 9, 10), dtype=np.float32)
    data[2:7, 3:8, 4:9] = 1
    data[3, 4, 5] = 2
    image = bb.load(
        _write_nifti(tmp_path / "seg.nii.gz", data),
        LUT="freesurfer",
        indices=(1,),
        interp="LANCZOS",
    )

    assert image.kind == "delineation"
    assert image.LUT is not None
    assert image.colormap is None
    assert 2 not in np.unique(image.volume.data)
    assert image.interp == "NEAREST"


def test_load_rejects_ambiguous_display_arguments(tmp_path):
    path = _write_nifti(tmp_path / "seg.nii.gz", np.ones((4, 4, 4), dtype=np.float32))

    with pytest.raises(ConfigError, match="LUT and colormap"):
        bb.load(path, LUT="freesurfer", colormap="gray")

    with pytest.raises(ConfigError, match="threshold"):
        bb.load(path, LUT="freesurfer", threshold=(0, 1))

    with pytest.raises(ConfigError, match="indices"):
        bb.load(path, indices=(1,))


def test_display_metadata_assignment_and_validation(tmp_path):
    intensity = _intensity_image(tmp_path)
    intensity.colormap = "viridis"
    intensity.threshold = (0, 50)
    intensity.scale = 2
    intensity.interp = "LINEAR"
    intensity.sharpen = 200
    intensity.crop_info = ((1, 5), (2, 7), (3, 8))

    assert intensity.colormap.name == "viridis"
    assert intensity.threshold == (0.0, 50.0)
    assert intensity.scale == 2.0
    assert intensity.interp == "BILINEAR"
    assert intensity.sharpen == 200
    assert intensity.crop_info == ((1, 5), (2, 7), (3, 8))

    delineation = _delineation_image(tmp_path)
    delineation.LUT = "freesurfer"
    delineation.interp = "LANCZOS"
    assert delineation.LUT is not None
    assert delineation.interp == "NEAREST"

    assert not hasattr(intensity, "change_LUT")
    assert not hasattr(intensity, "change_colormap")
    assert not hasattr(intensity, "apply_threshold")
    assert not hasattr(intensity, "apply_mask")
    with pytest.raises(ConfigError, match="colormap"):
        intensity.LUT = "freesurfer"
    with pytest.raises(ConfigError, match="LUT"):
        delineation.colormap = "gray"
    with pytest.raises(ConfigError, match="threshold"):
        delineation.threshold = (0, 1)
    with pytest.raises(ConfigError, match="threshold max"):
        intensity.threshold = (1, 0)
    with pytest.raises(ConfigError, match="crop_info"):
        intensity.crop_info = ((0, 100), (0, 2), (0, 2))
    with pytest.raises(AttributeError, match="Only display"):
        intensity.name = "changed"


def test_mask_assignment_updates_rendering_and_validates(tmp_path):
    image = _intensity_image(tmp_path)
    mask = np.ones(image.volume.data.shape, dtype=np.float16)
    mask[:4, :, :] = np.float16(0.5)

    image.mask = mask

    assert image.mask.dtype == np.float16
    assert image.mask.flags.writeable
    image.mask[0, 0, 0] = np.float16(0)
    assert image.mask[0, 0, 0] == np.float16(0)
    alpha = np.asarray(image.slice(5, plane="axial"))[..., 3]
    assert alpha.min() < 127
    assert 127 in alpha
    assert alpha.max() == 255

    with pytest.raises(ConfigError, match="same shape"):
        image.mask = np.ones((2, 2), dtype=np.float16)
    with pytest.raises(ConfigError, match="between 0 and 1"):
        image.mask = np.full(image.volume.data.shape, 2, dtype=np.float16)
    image.mask = None
    assert image.mask is None


def test_manual_crop_info_controls_display_without_changing_data(tmp_path):
    image = _intensity_image(tmp_path)
    original_shape = image.volume.data.shape

    image.crop_info = ((2, 7), (3, 8), (4, 9))

    assert image.volume.data.shape == original_shape
    assert image.crop_info == ((2, 7), (3, 8), (4, 9))
    assert image.slice(2, plane="axial").size == (15, 15)

    image.crop_info = ((1, 5), (2, 6), (3, 7))
    assert image.volume.data.shape == original_shape
    assert image.slice(1, plane="axial").size == (12, 12)

    image.crop_info = None
    assert image.crop_info is None
    assert image.slice(5, plane="axial").size == (24, 27)


def test_create_mask_returns_mutable_float16(monkeypatch, tmp_path):
    image = _intensity_image(tmp_path)
    expected = np.ones(image.volume.data.shape, dtype=np.float32)
    captured = {}

    def fake_create_mask_array(*args, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(image_module, "create_mask_array", fake_create_mask_array)
    mask = bb.create_mask(source=image, pad=2, low_thresh=1.5, max_thresh=99.0, cleanup=0)

    assert mask.dtype == np.float16
    assert mask.flags.writeable
    mask[0, 0, 0] = 0
    assert mask[0, 0, 0] == 0
    assert captured["pad"] == 2
    assert captured["low_thresh"] == 1.5
    assert captured["max_thresh"] == 99.0
    assert captured["cleanup"] == 0


def test_create_mask_enhances_by_default_and_can_be_disabled(monkeypatch, tmp_path):
    image = bb.load(
        _write_nifti(tmp_path / "large_mri.nii.gz", np.ones((21, 21, 21), dtype=np.float32))
    )
    hard_mask = np.zeros(image.volume.data.shape, dtype=np.float32)
    hard_mask[4:17, 4:17, 4:17] = 1
    hard_mask[8:13, 8:13, 8:13] = 0

    monkeypatch.setattr(image_module, "create_mask_array", lambda *args, **kwargs: hard_mask)

    enhanced = bb.create_mask(source=image)
    unchanged = bb.create_mask(source=image, enhance=False)

    assert enhanced.dtype == np.float16
    assert enhanced[10, 10, 10] > 0.5
    assert 0 < enhanced[3, 10, 10] < 1
    assert unchanged[10, 10, 10] == 0
    assert unchanged[4, 10, 10] == 1
    assert unchanged[3, 10, 10] == 0


def test_create_mask_array_uses_ants_options_and_pad(monkeypatch, tmp_path):
    image = _intensity_image(tmp_path)
    calls = {}

    class FakeMask:
        def __init__(self, array):
            self.array = array

        def numpy(self):
            return self.array

    class FakeAnts:
        def get_mask(self, ants_image, **kwargs):
            calls["get_mask"] = kwargs
            return FakeMask(np.ones(image.volume.data.shape, dtype=np.float32))

        def iMath(self, mask, operation, radius):
            calls["iMath"] = (operation, radius)
            return FakeMask(mask.numpy())

    monkeypatch.setattr(alignment_module, "_import_ants", lambda: FakeAnts())
    monkeypatch.setattr(
        alignment_module,
        "_ants_from_volume",
        lambda ants, volume, work_dir, name: object(),
    )

    mask = alignment_module.create_mask_array(
        image.volume,
        pad=3,
        low_thresh=2.0,
        max_thresh=50.0,
        cleanup=0,
    )

    assert mask.dtype == np.float16
    assert calls["get_mask"] == {"low_thresh": 2.0, "high_thresh": 50.0, "cleanup": 0}
    assert calls["iMath"] == ("MD", 3)


def test_create_mask_validates_options(tmp_path):
    image = _intensity_image(tmp_path)

    with pytest.raises(ConfigError, match="pad"):
        bb.create_mask(source=image, pad=-1)
    with pytest.raises(ConfigError, match="cleanup"):
        bb.create_mask(source=image, cleanup=-1)
    with pytest.raises(ConfigError, match="low_thresh"):
        bb.create_mask(source=image, low_thresh=np.inf)
    with pytest.raises(ConfigError, match="max_thresh"):
        bb.create_mask(source=image, max_thresh=np.inf)


def test_create_brainmask_returns_mutable_float16(monkeypatch, tmp_path):
    image = _intensity_image(tmp_path)
    expected = np.ones(image.volume.data.shape, dtype=np.float32)
    expected[0, :, :] = 0
    captured = {}

    def fake_create_brainmask_array(*args, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(image_module, "create_brainmask_array", fake_create_brainmask_array)
    mask = bb.create_brainmask(
        source=image,
        modality="t2",
        threshold=0.4,
        pad=2,
        cleanup=1,
    )

    assert mask.dtype == np.float16
    assert mask.flags.writeable
    assert mask[0, 0, 0] == 0
    mask[0, 0, 0] = 1
    assert mask[0, 0, 0] == 1
    assert captured["modality"] == "t2"
    assert captured["threshold"] == 0.4
    assert captured["pad"] == 2
    assert captured["cleanup"] == 1


def test_create_brainmask_array_uses_antspynet_threshold_cleanup_and_pad(monkeypatch, tmp_path):
    image = _intensity_image(tmp_path)
    probabilities = np.linspace(0, 1, image.volume.data.size, dtype=np.float32).reshape(
        image.volume.data.shape
    )
    calls = {"iMath": []}

    class FakeMask:
        def __init__(self, array):
            self.array = np.asarray(array, dtype=np.float32)

        def numpy(self):
            return self.array

    class FakeAnts:
        def threshold_image(self, probability, low_thresh, high_thresh):
            calls["threshold"] = (low_thresh, high_thresh)
            return FakeMask((probability.numpy() >= low_thresh).astype(np.float32))

        def iMath(self, mask, operation, *args):
            calls["iMath"].append((operation, args))
            return mask

    def fake_brain_extraction(ants_image, *, modality, verbose):
        calls["brain_extraction"] = (ants_image, modality, verbose)
        return FakeMask(probabilities)

    monkeypatch.setattr(alignment_module, "_import_ants", lambda: FakeAnts())
    monkeypatch.setattr(
        alignment_module,
        "_import_antspynet_brain_extraction",
        lambda: fake_brain_extraction,
    )
    monkeypatch.setattr(
        alignment_module,
        "_ants_from_volume",
        lambda ants, volume, work_dir, name: "ants-image",
    )

    mask = alignment_module.create_brainmask_array(
        image.volume,
        modality="flair",
        threshold=0.7,
        pad=2,
        cleanup=1,
        debug=True,
    )

    assert mask.dtype == np.float16
    assert set(np.unique(mask)) <= {np.float16(0), np.float16(1)}
    assert calls["brain_extraction"] == ("ants-image", "flair", True)
    assert calls["threshold"] == (0.7, 1.0)
    assert calls["iMath"] == [
        ("ME", (1,)),
        ("GetLargestComponent", ()),
        ("MD", (1,)),
        ("FillHoles", ()),
        ("MD", (2,)),
    ]


def test_ants_compatibility_installs_one_hot_helpers():
    class FakeAntsModule:
        pass

    module = FakeAntsModule()

    def fake_from_numpy_like(data, domain_image):
        return {"data": np.asarray(data), "domain": domain_image}

    alignment_module._install_ants_compatibility(
        module,
        from_numpy_like=fake_from_numpy_like,
    )

    segmentation = np.array([[0, 1], [2, 1]], dtype=np.int16)
    one_hot = module.segmentation_to_one_hot(segmentation, segmentation_labels=(0, 1, 2))
    assert one_hot.shape == (2, 2, 3)
    assert np.array_equal(one_hot[..., 1], segmentation == 1)

    probability_images = module.one_hot_to_segmentation(one_hot, "domain-image")
    assert len(probability_images) == 3
    assert probability_images[1]["domain"] == "domain-image"
    assert np.array_equal(probability_images[1]["data"], one_hot[..., 1])

    channel_first = one_hot.transpose((2, 0, 1))
    probability_images = module.one_hot_to_segmentation(
        channel_first,
        "domain-image",
        channel_first_ordering=True,
    )
    assert np.array_equal(probability_images[2]["data"], one_hot[..., 2])


def test_create_brainmask_validates_options(tmp_path):
    image = _intensity_image(tmp_path)

    with pytest.raises(ConfigError, match="modality"):
        bb.create_brainmask(source=image, modality="")
    with pytest.raises(ConfigError, match="threshold"):
        bb.create_brainmask(source=image, threshold=0)
    with pytest.raises(ConfigError, match="threshold"):
        bb.create_brainmask(source=image, threshold=1)
    with pytest.raises(ConfigError, match="pad"):
        bb.create_brainmask(source=image, pad=-1)
    with pytest.raises(ConfigError, match="cleanup"):
        bb.create_brainmask(source=image, cleanup=-1)


def test_crop_tracks_and_replaces_crop_info_without_cropping_volume(monkeypatch, tmp_path):
    image = _intensity_image(tmp_path)
    mask = np.ones(image.volume.data.shape, dtype=np.float16)
    mask[:4, :, :] = np.float16(0.5)
    image.mask = mask
    first_crop_info = ((2, 7), (3, 8), (4, 9))
    second_crop_info = ((1, 5), (2, 6), (3, 7))
    seen_shapes = []

    def fake_crop_volume(volume, **kwargs):
        seen_shapes.append(volume.data.shape)
        crop_info = (
            tuple(tuple(item) for item in kwargs["crop_info"])
            if kwargs["crop_info"]
            else first_crop_info
        )
        slices = tuple(slice(start, stop) for start, stop in crop_info)
        data = volume.data[slices]
        cropped_volume = volume_from_data(volume.path, data, volume.affine)
        return AntsCropResult(volume=cropped_volume, crop_info=crop_info)

    monkeypatch.setattr(image_module, "crop_volume", fake_crop_volume)
    cropped = bb.crop(image)

    assert cropped.crop_info == first_crop_info
    assert cropped.volume.data.shape == image.volume.data.shape
    assert cropped.mask.shape == image.mask.shape
    assert cropped.slice(2, plane="axial").size == (15, 15)
    assert np.all(cropped.mask[:4] == np.float16(0.5))

    recropped = bb.crop(cropped, crop_info=second_crop_info)
    assert recropped.crop_info == second_crop_info
    assert recropped.volume.data.shape == image.volume.data.shape
    assert recropped.slice(1, plane="axial").size == (12, 12)
    assert seen_shapes == [image.volume.data.shape, image.volume.data.shape]


def test_crop_pending_transform_uses_transformed_volume(monkeypatch, tmp_path):
    image = _intensity_image(tmp_path)
    image.mask = np.ones(image.volume.data.shape, dtype=np.float16)
    transformed_data = np.full((6, 7, 8), 42, dtype=np.float32)
    transformed_mask = np.full((6, 7, 8), 0.5, dtype=np.float32)
    calls = {"apply": 0, "crop_shape": None, "crop_value": None}
    crop_info = ((1, 5), (2, 6), (3, 7))

    def fake_register_ants_transform(**kwargs):
        return AntsRegistrationResult(
            forward_transforms=(AntsTransformFile("rigid.mat", b"rigid"),),
            inverse_transforms=(),
        )

    def fake_apply_ants_transform(**kwargs):
        calls["apply"] += 1
        moving_data = np.asarray(kwargs["moving"].data)
        data = transformed_mask if np.allclose(moving_data, 1.0) else transformed_data
        return image_from_volume_data(kwargs["fixed"], data=data)

    def fake_crop_volume(volume, **kwargs):
        calls["crop_shape"] = volume.data.shape
        calls["crop_value"] = float(volume.data[0, 0, 0])
        slices = tuple(slice(start, stop) for start, stop in crop_info)
        cropped_volume = volume_from_data(volume.path, volume.data[slices], volume.affine)
        return AntsCropResult(volume=cropped_volume, crop_info=crop_info)

    monkeypatch.setattr(image_module, "register_ants_transform", fake_register_ants_transform)
    monkeypatch.setattr(image_module, "apply_ants_transform", fake_apply_ants_transform)
    monkeypatch.setattr(image_module, "crop_volume", fake_crop_volume)

    pending = bb.transform(source=image, target=image, warp="Rigid")
    cropped = bb.crop(pending)

    assert calls == {"apply": 2, "crop_shape": transformed_data.shape, "crop_value": 42.0}
    assert cropped.transform_info is not None
    assert cropped.transform_info.applied is True
    assert cropped.crop_info == crop_info
    assert cropped.volume.data.shape == transformed_data.shape
    assert cropped.mask.shape == transformed_data.shape
    assert np.all(cropped.mask == np.float16(0.5))
    assert cropped.slice(1, plane="axial").size == (12, 12)


def test_smooth_image_validates_kernel_and_preserves_mask(tmp_path):
    image = _intensity_image(tmp_path)
    mask = np.ones(image.volume.data.shape, dtype=np.float16)
    image.mask = mask

    smoothed = bb.smooth_image(source=image, kernel=1)

    assert smoothed.kind == "intensity"
    assert smoothed.mask.shape == image.mask.shape

    with pytest.raises(ConfigError, match="non-negative"):
        bb.smooth_image(source=image, kernel=-1)
    with pytest.raises(ConfigError, match="non-negative"):
        bb.smooth_image(source=image, kernel=(1, -1, 1))
    with pytest.raises(ConfigError, match="length"):
        bb.smooth_image(source=image, kernel=(1, 1))
    with pytest.raises(ConfigError):
        bb.smooth_image(source=_delineation_image(tmp_path), kernel=1)


def test_smooth_mask_accepts_mask_matrix_and_validates_options():
    mask = np.zeros((7, 7, 7), dtype=np.float16)
    mask[3, 3, 3] = 1

    smoothed = bb.smooth_mask(source=mask, kernel=1)

    assert smoothed.dtype == np.float16
    assert smoothed.shape == mask.shape
    assert smoothed.flags.writeable
    assert 0 <= smoothed.min() <= smoothed.max() <= 1
    assert smoothed[3, 3, 3] < 1
    assert smoothed[3, 3, 3] > smoothed[0, 0, 0]

    smoothed_with_affine = bb.smooth_mask(mask=mask, kernel=(1, 1, 1), affine=np.eye(4))
    assert smoothed_with_affine.shape == mask.shape

    with pytest.raises(ConfigError, match="either source= or mask="):
        bb.smooth_mask(source=mask, mask=mask, kernel=1)
    with pytest.raises(ConfigError, match="requires"):
        bb.smooth_mask(kernel=1)
    with pytest.raises(ConfigError, match="3D"):
        bb.smooth_mask(source=np.ones((3, 3), dtype=np.float16), kernel=1)
    with pytest.raises(ConfigError, match="between 0 and 1"):
        bb.smooth_mask(source=np.full((3, 3, 3), 2, dtype=np.float16), kernel=1)
    with pytest.raises(ConfigError, match="non-negative"):
        bb.smooth_mask(source=mask, kernel=-1)
    with pytest.raises(ConfigError, match="length"):
        bb.smooth_mask(source=mask, kernel=(1, 1))
    with pytest.raises(ConfigError, match="affine"):
        bb.smooth_mask(source=mask, kernel=1, affine=np.eye(3))


def test_old_smooth_name_is_not_public():
    assert not hasattr(bb, "smooth")


def test_save_load_bbi_preserves_rendering_metadata_without_rgba(tmp_path):
    image = bb.load(
        _write_nifti(tmp_path / "mri.nii.gz", np.ones((8, 9, 10), dtype=np.float32)),
        scale=2,
        interp="LANCZOS",
        sharpen=20,
    )
    image.threshold = (0, 100)
    path = image.save(tmp_path / "image.bbi")

    with np.load(path, allow_pickle=False) as archive:
        assert "rgba" not in archive.files

    loaded = bb.load(path)

    assert loaded.kind == "intensity"
    assert loaded.scale == 2.0
    assert loaded.interp == "LANCZOS"
    assert loaded.sharpen == 20
    assert loaded.slice(5, plane="axial").size == (16, 18)


def test_save_load_bbi_embeds_lazy_transform_template_geometry(monkeypatch, tmp_path):
    source = _intensity_image(tmp_path)
    target_path = _write_nifti(
        tmp_path / "target.nii.gz",
        np.ones((5, 6, 7), dtype=np.float32),
    )
    target = bb.load(target_path)

    def fake_register_ants_transform(**kwargs):
        return AntsRegistrationResult(
            forward_transforms=(AntsTransformFile("rigid.mat", b"rigid"),),
            inverse_transforms=(),
        )

    monkeypatch.setattr(image_module, "register_ants_transform", fake_register_ants_transform)

    transformed = bb.transform(source=source, target=target, warp="Rigid")
    path = transformed.save(tmp_path / "pending.bbi")

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        transform_metadata = metadata["transform_info"]
        assert "template_data" not in archive.files
        assert transform_metadata["template_shape"] == list(target.volume.data.shape)
        assert transform_metadata["template_affine"] == target.volume.affine.tolist()

    target_path.unlink()
    loaded = bb.load(path)

    assert loaded.transform_info is not None
    assert loaded.transform_info.template_volume is None
    assert loaded.transform_info.template_shape == target.volume.data.shape
    assert np.array_equal(loaded.transform_info.template_affine, target.volume.affine)

    def fake_apply_ants_transform(**kwargs):
        assert kwargs["fixed"].data.shape == target.volume.data.shape
        assert np.all(kwargs["fixed"].data == 0)
        return image_from_volume_data(kwargs["fixed"], data=np.full((5, 6, 7), 3))

    monkeypatch.setattr(image_module, "apply_ants_transform", fake_apply_ants_transform)

    resliced = loaded.reslice()

    assert resliced.volume.data.shape == target.volume.data.shape
    assert np.all(resliced.volume.data == 3)


def test_transform_and_apply_transform_are_lazy_and_preserve_mask(monkeypatch, tmp_path):
    from PIL import Image

    image = _intensity_image(tmp_path)
    image.mask = np.ones((8, 9, 10), dtype=np.float16)
    applied = {"count": 0, "interp": []}

    def fake_register_ants_transform(**kwargs):
        return AntsRegistrationResult(
            forward_transforms=(AntsTransformFile("rigid.mat", b"rigid"),),
            inverse_transforms=(),
        )

    def fake_apply_ants_transform(**kwargs):
        applied["count"] += 1
        applied["interp"].append(kwargs["interp"])
        moving_data = np.asarray(kwargs["moving"].data)
        value = 0.5 if np.allclose(moving_data, 1.0) else 42
        data = np.full_like(moving_data, value, dtype=np.float32)
        return image_from_volume_data(kwargs["moving"], data=data)

    monkeypatch.setattr(image_module, "register_ants_transform", fake_register_ants_transform)
    monkeypatch.setattr(image_module, "apply_ants_transform", fake_apply_ants_transform)

    transformed = bb.transform(source=image, target=image, warp="Rigid")

    assert transformed.mask is not None
    assert np.array_equal(transformed.mask, image.mask)
    assert transformed.transform_info is not None
    assert transformed.transform_info.applied is False
    assert np.array_equal(transformed.volume.data, image.volume.data)
    assert applied["count"] == 0

    rendered_pending = transformed.slice(5, plane="axial")
    assert rendered_pending.mode == "RGBA"
    assert applied["count"] == 2

    resliced = transformed.reslice(interp=Image.Resampling.LANCZOS)
    assert applied["count"] == 4
    assert applied["interp"] == [
        "bSpline",
        "bSpline",
        "lanczosWindowedSinc",
        "lanczosWindowedSinc",
    ]
    assert resliced.transform_info is not None
    assert resliced.transform_info.applied is True
    assert np.all(resliced.volume.data == 42)
    assert resliced.mask is not None
    assert resliced.mask.dtype == np.float16
    assert np.all(resliced.mask == np.float16(0.5))

    rendered = resliced.slice(5, plane="axial")
    assert rendered.size == (24, 27)

    record = image_module.TransformRecord(
        forward_transforms=(AntsTransformFile("rigid.mat", b"rigid"),),
        template_volume=image.volume,
    )
    pending = bb.apply_transform(source=image, matrix=record)

    assert pending.mask is not None
    assert np.array_equal(pending.mask, image.mask)
    assert pending.transform_info is not None
    assert pending.transform_info.applied is False
    assert np.array_equal(pending.volume.data, image.volume.data)


def test_transform_return_transform(monkeypatch, tmp_path):
    image = _intensity_image(tmp_path)
    image.mask = np.ones((8, 9, 10), dtype=np.float16)

    def fake_register_ants_transform(**kwargs):
        return AntsRegistrationResult(
            forward_transforms=(AntsTransformFile("rigid.mat", b"rigid"),),
            inverse_transforms=(),
        )

    monkeypatch.setattr(image_module, "register_ants_transform", fake_register_ants_transform)

    transform = bb.transform(source=image, target=image, warp="Rigid", return_transform=True)

    assert isinstance(transform, image_module.TransformRecord)
    assert transform.applied is False


def test_transformed_slice_reslices_temporarily(monkeypatch, tmp_path):
    image = _intensity_image(tmp_path)
    calls = {"apply": 0}

    def fake_register_ants_transform(**kwargs):
        return AntsRegistrationResult(
            forward_transforms=(AntsTransformFile("affine.mat", b"affine"),),
            inverse_transforms=(),
        )

    def fake_apply_ants_transform(**kwargs):
        calls["apply"] += 1
        data = np.full_like(kwargs["moving"].data, 25, dtype=np.float32)
        return image_from_volume_data(kwargs["moving"], data=data)

    monkeypatch.setattr(image_module, "register_ants_transform", fake_register_ants_transform)
    monkeypatch.setattr(image_module, "apply_ants_transform", fake_apply_ants_transform)

    transformed = bb.transform(source=image, target=image, warp="Affine")

    first = transformed.slice(5, plane="axial")
    second = transformed.slice(6, plane="axial")

    assert first.mode == "RGBA"
    assert second.mode == "RGBA"
    assert calls["apply"] == 2


def test_slice_uses_display_metadata_and_validates_options(monkeypatch, tmp_path):
    from PIL import Image

    image = bb.load(
        _write_nifti(tmp_path / "display.nii.gz", np.ones((8, 9, 10), dtype=np.float32)),
        scale=2.5,
        interp=Image.Resampling.BICUBIC,
        sharpen=200,
    )
    calls = {"interp": []}
    transformed_data = np.zeros_like(image.volume.data, dtype=np.float32)
    transformed_data[4, 4, 4] = 10

    def fake_register_ants_transform(**kwargs):
        return AntsRegistrationResult(
            forward_transforms=(AntsTransformFile("affine.mat", b"affine"),),
            inverse_transforms=(),
        )

    def fake_apply_ants_transform(**kwargs):
        calls["interp"].append(kwargs["interp"])
        return image_from_volume_data(kwargs["moving"], data=transformed_data)

    monkeypatch.setattr(image_module, "register_ants_transform", fake_register_ants_transform)
    monkeypatch.setattr(image_module, "apply_ants_transform", fake_apply_ants_transform)

    assert image.slice(5, plane="axial").size == (20, 22)
    assert image.scale == 2.5
    assert image.interp == "BICUBIC"
    assert image.sharpen == 200
    copied = image.reslice("BICUBIC")
    assert copied is not image
    assert copied.interp == image.interp
    assert np.array_equal(copied.volume.data, image.volume.data)
    with pytest.raises(ConfigError, match="ANTs interpolation"):
        image.reslice("bad")

    transformed = bb.transform(source=image, target=image, warp="Affine")
    resliced = transformed.reslice("LANCZOS")

    assert calls["interp"] == ["lanczosWindowedSinc"]
    assert np.array_equal(resliced.volume.data, transformed_data)
    assert resliced.interp == "BICUBIC"
    assert resliced.slice(4, plane="axial").size == (20, 22)

    with pytest.raises(ConfigError, match="sharpen"):
        bb.load(_write_nifti(tmp_path / "bad_sharpen1.nii.gz", image.volume.data), sharpen=-1)
    with pytest.raises(ConfigError, match="sharpen"):
        bb.load(_write_nifti(tmp_path / "bad_sharpen2.nii.gz", image.volume.data), sharpen=1.5)
    with pytest.raises(ConfigError, match="scale"):
        bb.load(_write_nifti(tmp_path / "bad_scale1.nii.gz", image.volume.data), scale=0)
    with pytest.raises(ConfigError, match="scale"):
        bb.load(_write_nifti(tmp_path / "bad_scale2.nii.gz", image.volume.data), scale=np.inf)
    with pytest.raises(ConfigError, match="interp"):
        bb.load(_write_nifti(tmp_path / "bad_interp.nii.gz", image.volume.data), interp="bad")
    with pytest.raises(ConfigError, match="intensity"):
        bb.load(
            _write_nifti(tmp_path / "bad_label.nii.gz", image.volume.data),
            LUT="freesurfer",
            sharpen=1,
        )


def test_slice_interp_override_does_not_mutate_metadata(monkeypatch, tmp_path):
    image = bb.load(
        _write_nifti(tmp_path / "override.nii.gz", np.ones((8, 9, 10), dtype=np.float32)),
        scale=2,
        interp="BICUBIC",
    )
    image.mask = np.ones(image.volume.data.shape, dtype=np.float16)
    calls = []

    def fake_resize(array, *, output_shape, interp):
        calls.append(interp)
        return np.zeros(output_shape, dtype=np.float32)

    monkeypatch.setattr(image_module, "_resize_2d_array", fake_resize)

    rendered = image.slice(5, plane="axial", interp="NEAREST")

    assert rendered.mode == "RGBA"
    assert calls == ["NEAREST", "NEAREST"]
    assert image.interp == "BICUBIC"

    with pytest.raises(ConfigError, match="interp"):
        image.slice(5, plane="axial", interp="bad")
    assert image.interp == "BICUBIC"


def test_colorbar_and_figure(tmp_path):
    bar = bb.colorbar(colormap="gray", height=8, length=16)
    assert bar.mode == "RGBA"
    assert bar.size == (16, 8)

    image = _intensity_image(tmp_path)
    figure = bb.bbfigure(size=np.sqrt(2), grid=(1, 1), dpi=20, background=(0, 0, 0, 0))
    figure[0] = image.slice(5)
    rendered = figure.render()

    assert rendered.mode == "RGBA"
    assert rendered.size == (20, 20)
    assert not hasattr(figure, "show")

    grid_ratio = bb.bbfigure(size=5, grid=(2, 3), dpi=10)
    assert grid_ratio.size == 5.0
    assert grid_ratio.grid == (2, 3)
    assert grid_ratio.render().size == (28, 42)

    wide_panels = bb.bbfigure(size=np.sqrt(10), grid=(2, 1), dpi=10, panel_aspect=1.5)
    assert wide_panels.panel_aspect == 1.5
    assert wide_panels.render().size == (30, 10)

    wide_alias = bb.bbfigure(size=np.sqrt(10), grid=(2, 1), dpi=10, box_ratio=1.5)
    assert wide_alias.panel_aspect == 1.5
    assert wide_alias.render().size == (30, 10)

    one_dimensional = bb.bbfigure(size=np.sqrt(5), grid=2, dpi=10)
    assert one_dimensional.grid == (2, 1)
    assert one_dimensional.render().size == (20, 10)
    one_dimensional[0] = image.slice(4)
    one_dimensional[1] = image.slice(5)

    two_dimensional = bb.bbfigure(size=np.sqrt(8), grid=(2, 2), dpi=10)
    two_dimensional[0, 0] = image.slice(4)
    two_dimensional[1, 0] = image.slice(5)
    two_dimensional[0, 1] = image.slice(6)
    two_dimensional[1, 1] = image.slice(7)
    assert two_dimensional.render().size == (20, 20)

    with pytest.raises(ValueError, match="panel_aspect"):
        bb.bbfigure(size=1, grid=(1, 1), panel_aspect=0)
    with pytest.raises(ValueError, match="panel_aspect or box_ratio"):
        bb.bbfigure(size=1, grid=(1, 1), panel_aspect=2, box_ratio=1.5)


def test_figure_preserves_shared_visual_scale_by_default():
    from PIL import Image

    def visible_size(image):
        alpha = np.asarray(image)[..., 3]
        rows, cols = np.nonzero(alpha)
        return (int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1))

    large = Image.new("RGBA", (80, 100), (255, 0, 0, 255))
    small = Image.new("RGBA", (40, 20), (0, 255, 0, 255))

    preserved = bb.bbfigure(
        size=np.sqrt(5),
        grid=(2, 1),
        dpi=100,
        background=(0, 0, 0, 0),
    )
    preserved[0] = large
    preserved[1] = small
    rendered = preserved.render()

    assert preserved.preserve_scale is True
    assert rendered.size == (200, 100)
    assert visible_size(rendered.crop((0, 0, 100, 100))) == (80, 100)
    assert visible_size(rendered.crop((100, 0, 200, 100))) == (40, 20)

    independent = bb.bbfigure(
        size=np.sqrt(5),
        grid=(2, 1),
        dpi=100,
        background=(0, 0, 0, 0),
        preserve_scale=False,
    )
    independent[0] = large
    independent[1] = small
    independently_rendered = independent.render()

    assert visible_size(independently_rendered.crop((0, 0, 100, 100))) == (80, 100)
    assert visible_size(independently_rendered.crop((100, 0, 200, 100))) == (100, 50)

    with pytest.raises(ValueError, match="preserve_scale"):
        bb.bbfigure(size=1, grid=(1, 1), preserve_scale=1)


def test_notebook_repr_does_not_dump_matrices(tmp_path):
    image = _intensity_image(tmp_path)
    figure = bb.bbfigure(size=np.sqrt(2), grid=(1, 1), dpi=20)
    figure[0] = image.slice(5)

    image_repr = repr(image)
    figure_repr = repr(figure)

    assert "BBImage(" in image_repr
    assert "volume=" not in image_repr
    assert "rgba=" not in image_repr
    assert "array(" not in image_repr
    assert "BBFigure(" in figure_repr
    assert "panels={" not in figure_repr
    assert "layers=" not in figure_repr


def test_metadata_prints_shapes_not_matrices(tmp_path, capsys):
    image = _intensity_image(tmp_path)
    image.mask = np.ones((8, 9, 10), dtype=np.float16)

    result = image.metadata()
    output = capsys.readouterr().out

    assert result is None
    assert "BBImage metadata" in output
    assert "data: matrix shape=(8, 9, 10), dtype=float32" in output
    assert "affine: matrix shape=(4, 4)" in output
    assert "mask: matrix shape=(8, 9, 10), dtype=float16" in output
    assert "array(" not in output


def test_generated_colormap_assets_load():
    names = (
        "grey",
        "viridis",
        "plasma",
        "greys",
        "jet",
        "rainbow",
        "coolwarm",
        "bwr",
        "seismic",
        "bone",
        "gray",
        "hot",
        "copper",
        "magma",
        "reds",
        "blues",
        "nih",
    )

    for name in names:
        colors = colormap_module.load_colormap(name)
        assert colors.shape == (256, 4)
        assert np.all((colors >= 0) & (colors <= 1))

        soft_colors = colormap_module.load_colormap(f"soft-{name}")
        expected_alpha = np.ones(256, dtype=np.float32)
        expected_alpha[0] = 0
        expected_alpha[1:5] = np.sqrt(np.linspace(0, 1, 4, dtype=np.float32))

        assert soft_colors.shape == (256, 4)
        assert np.allclose(soft_colors[:, :3], colors[:, :3])
        assert np.allclose(soft_colors[:, 3], expected_alpha)

    asset_dir = colormap_module.builtin_colormap_path("gray").parent
    assert not (asset_dir / "grey.csv").exists()
    assert not (asset_dir / "nih_pet.csv").exists()

    with pytest.raises(ConfigError, match="Unknown built-in colormap"):
        colormap_module.load_colormap("nih_pet")


def test_colormap_file_paths_load_uniformly(tmp_path):
    path = tmp_path / "custom.csv"
    np.savetxt(path, np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.float32), delimiter=",")

    from_path = colormap_module.load_colormap(path)
    from_string = colormap_module.load_colormap(str(path))
    resolved_from_path = colormap_module.resolve_colormap(path)
    resolved_from_string = colormap_module.resolve_colormap(str(path))

    assert np.allclose(from_path, from_string)
    assert resolved_from_path.name == "custom"
    assert resolved_from_string.name == "custom"

    with pytest.raises(ConfigError, match="Colormap file does not exist"):
        colormap_module.load_colormap(tmp_path / "missing.csv")


def test_make_video_saves_pillow_animation_formats(tmp_path):
    from PIL import Image, features

    frames = _animation_figures()
    gif_path = bb.make_video(frames=frames, timing=0.1, filepath=tmp_path / "movie.gif")
    png_path = bb.make_video(frames=frames, timing=0.1, filepath=tmp_path / "movie.png")
    apng_alias_path = bb.make_video(frames=frames, timing=0.1, filepath=tmp_path / "movie.apgn")

    assert gif_path.exists()
    assert png_path.exists()
    assert apng_alias_path.exists()

    with Image.open(gif_path) as image:
        assert image.n_frames == len(frames)

    with Image.open(png_path) as image:
        assert image.n_frames == len(frames)

    if features.check("webp"):
        webp_path = bb.make_video(frames=frames, timing=0.1, filepath=tmp_path / "movie.webp")
        assert webp_path.exists()


def test_make_video_rejects_bbimage_slices(tmp_path):
    image = _intensity_image(tmp_path)

    with pytest.raises(ConfigError, match="BBFigure"):
        bb.make_video(
            frames=[image.slice(4, plane="axial"), image.slice(5, plane="axial")],
            timing=0.1,
            filepath=tmp_path / "slices.gif",
        )


def test_make_video_accepts_bbfigures(tmp_path):
    image = _intensity_image(tmp_path)
    frames = []
    for index in (4, 5):
        figure = bb.bbfigure(size=np.sqrt(2), grid=(1, 1), dpi=20, background=(0, 0, 0, 0))
        figure[0] = image.slice(index, plane="axial")
        frames.append(figure)

    output = bb.make_video(frames=frames, timing=0.1, filepath=tmp_path / "figures.gif")

    assert output.exists()


def test_make_video_validates_inputs(tmp_path):
    frames = _animation_figures()
    wide = bb.bbfigure(size=np.sqrt(5), grid=(2, 1), dpi=10)

    with pytest.raises(ConfigError, match="extensions"):
        bb.make_video(frames=frames, timing=0.1, filepath=tmp_path / "movie.mov")
    with pytest.raises(ConfigError, match="positive"):
        bb.make_video(frames=frames, timing=0, filepath=tmp_path / "movie.gif")
    with pytest.raises(ConfigError, match="at least one"):
        bb.make_video(frames=[], timing=0.1, filepath=tmp_path / "movie.gif")
    with pytest.raises(ConfigError, match="same pixel"):
        bb.make_video(
            frames=[frames[0], wide],
            timing=0.1,
            filepath=tmp_path / "movie.gif",
        )
    with pytest.raises(ConfigError, match="BBFigure"):
        bb.make_video(frames=_animation_frames(), timing=0.1, filepath=tmp_path / "movie.gif")


def test_make_video_writes_mp4_with_optional_writer(monkeypatch, tmp_path):
    written = []
    seen = {}

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def append_data(self, array):
            written.append(array)

    def fake_writer_factory():
        def factory(filepath, *, fps):
            seen["filepath"] = filepath
            seen["fps"] = fps
            return FakeWriter()

        return factory

    monkeypatch.setattr(figure_module, "_imageio_writer_factory", fake_writer_factory)

    output = bb.make_video(
        frames=_animation_figures(), timing=0.25, filepath=tmp_path / "movie.mp4"
    )

    assert output == tmp_path / "movie.mp4"
    assert seen["filepath"] == output
    assert seen["fps"] == 4.0
    assert len(written) == 3
    assert written[0].shape == (10, 10, 3)


def test_make_video_reports_missing_mp4_dependencies(monkeypatch, tmp_path):
    def missing_writer_factory():
        raise DependencyError("Install imageio and imageio-ffmpeg.")

    monkeypatch.setattr(figure_module, "_imageio_writer_factory", missing_writer_factory)

    with pytest.raises(DependencyError, match="imageio"):
        bb.make_video(frames=_animation_figures(), timing=0.1, filepath=tmp_path / "movie.mp4")


def test_bbi_load_rejects_processing_options(tmp_path):
    image = _intensity_image(tmp_path)
    path = image.save(tmp_path / "saved.bbi")

    with pytest.raises(ConfigError, match="BBI archives"):
        bb.load(path, colormap="viridis")


def test_no_render_figure_public_api():
    assert not hasattr(bb, "render_figure")
    assert not hasattr(bb, "templates")


def test_bbi_kind_validation(tmp_path):
    image = _intensity_image(tmp_path)
    path = image.save(tmp_path / "bad.bbi")

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["kind"] = "mask"
    arrays["metadata"] = np.array(json.dumps(metadata))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    with pytest.raises(ConfigError, match="Unsupported BBI image kind"):
        bb.load(path)
