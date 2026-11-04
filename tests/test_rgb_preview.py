"""Independent BT.601 limited-range fixtures; no learning-space conversion."""

import numpy as np
import pytest

from cognio.gui import ycbcr_rgb


def model_units(codes):
    values = np.array(codes, dtype=np.float32)
    values[..., 1:] -= 128
    return np.moveaxis(values / 255, -1, 0)


def test_neutral_chroma_uses_limited_not_full_range():
    codes = [[[y, 128, 128] for y in (0, 16, 71, 126, 180, 235, 255)]]
    expected = np.repeat(np.array([[[0], [0], [64], [128], [191], [255], [255]]], np.uint8), 3, axis=2)
    np.testing.assert_array_equal(ycbcr_rgb(model_units(codes)), expected)


def test_primary_code_samples_component_order_layout_and_ownership():
    # Rounded standard limited-range BT.601 primary samples, arranged asymmetrically.
    codes = [[[81, 90, 240], [145, 54, 34], [41, 240, 110]],
             [[235, 128, 128], [16, 128, 128], [126, 128, 128]]]
    expected = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                         [[255, 255, 255], [0, 0, 0], [128, 128, 128]]])
    values = model_units(codes)[:, :, ::-1]  # Noncontiguous CHW is also valid.
    before = values.copy()
    values.flags.writeable = False
    rgb = ycbcr_rgb(values)
    np.testing.assert_allclose(rgb, expected[:, ::-1], atol=1, rtol=0)
    assert rgb.shape == (2, 3, 3) and rgb.dtype == np.uint8 and rgb.flags.c_contiguous
    assert not np.shares_memory(rgb, values)
    np.testing.assert_array_equal(values, before)


def test_fractional_resampled_measurements_round_trip_primary_and_mixed_colors():
    expected = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                         [[64, 128, 192], [224, 96, 32], [51, 51, 51]]], dtype=np.uint8)
    r, g, b = np.moveaxis(expected.astype(np.float64)/255, -1, 0)
    y = .299*r + .587*g + .114*b
    cb = (b-y)/1.772
    cr = (r-y)/1.402
    values = np.stack(((16+219*y)/255, 224*cb/255, 224*cr/255)).astype(np.float32)
    np.testing.assert_array_equal(ycbcr_rgb(values), expected)
    # Every component contributes, including luminance and both chroma planes.
    for component in range(3):
        changed = values.copy()
        changed[component] = 0
        assert not np.array_equal(ycbcr_rgb(changed), expected)


def test_extremes_clip_rgb_after_conversion_and_invalid_pixels_are_visible():
    values = np.array([[[2, -1, .5, .5, np.finfo(np.float32).max]],
                       [[0, 0, 2, -2, 0]], [[0, 0, -2, 2, 0]]], dtype=np.float32)
    before = values.copy()
    with np.errstate(all="raise"):
        rgb = ycbcr_rgb(values)
    np.testing.assert_array_equal(rgb, [[[255,255,255], [0,0,0], [0,255,255], [255,0,0], [255,255,255]]])
    np.testing.assert_array_equal(values, before)
    # Clamping code Y before conversion would incorrectly produce a brighter red.
    np.testing.assert_array_equal(ycbcr_rgb(np.array([[[-.1]], [[0]], [[.3]]])), [[[74,0,0]]])
    invalid = values.copy()
    invalid[0,0,0], invalid[1,0,1], invalid[2,0,2] = np.nan, np.inf, -np.inf
    with np.errstate(all="raise"):
        rgb = ycbcr_rgb(invalid)
    np.testing.assert_array_equal(rgb[0,:3], [[255,0,255]]*3)


@pytest.mark.parametrize("shape", [(128,128), (128,128,3), (1,2,3)])
def test_preview_requires_three_chw_planes(shape):
    with pytest.raises(ValueError, match="CHW"):
        ycbcr_rgb(np.zeros(shape, dtype=np.float32))
