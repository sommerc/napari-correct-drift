import numpy as np

from napari_correct_drift._core import ISTabilizer, ROIRect
from napari_correct_drift._sample_data import Example_TYX, Example_TZYX


def test3d():
    example = Example_TZYX()
    img_3d = example.create()
    ist_3d = ISTabilizer(img_3d, "tzyx")

    for t0 in [0, 16]:
        ofs = ist_3d.estimate_shifts_absolute(t0)
        okay = (ofs - example.shifts + example.shifts[t0]) % img_3d.shape[-3:]
        assert np.allclose(okay, 0), str(okay)


def test2d():
    example = Example_TYX()
    img_2d = example.create()
    ist_2d = ISTabilizer(img_2d, "tyx")

    for t0 in [0, 17]:
        ofs = ist_2d.estimate_shifts_absolute(t0)
        okay = (ofs - example.shifts + example.shifts[t0]) % (
            (1,) + img_2d.shape[-2:]
        )
        assert np.allclose(okay, 0), str(okay)


def test2d_C():
    example = Example_TYX()
    img_2d = example.create()
    img_2d = np.stack([img_2d, img_2d // 2], axis=1)
    ist_2d = ISTabilizer(img_2d, "tcyx")

    for t0 in [0, 7]:
        ofs = ist_2d.estimate_shifts_absolute(t0, channel=0)
        okay = (ofs - example.shifts + example.shifts[t0]) % (
            (1,) + img_2d.shape[-2:]
        )
        assert np.allclose(okay, 0), str(okay)


def test3d_C():
    example = Example_TZYX()
    img_3d = example.create()

    img_3d = np.stack([img_3d, img_3d // 2], axis=1)

    ist_3d = ISTabilizer(img_3d, "tczyx")

    for t0 in [0, 16]:
        ofs = ist_3d.estimate_shifts_absolute(t0, channel=0)
        okay = (ofs - example.shifts + example.shifts[t0]) % img_3d.shape[-3:]
        assert np.allclose(okay, 0), str(okay)


def test2d_roi():
    example = Example_TYX()
    img_2d = example.create()
    ist_2d = ISTabilizer(img_2d, "tyx")

    poly2d = np.array(
        [
            [23.0, 122.57642757, 142.27071085],
            [23.0, 122.57642757, 190.77701484],
            [23.0, 181.65929409, 190.77701484],
            [23.0, 181.65929409, 142.27071085],
        ]
    )

    roi = ROIRect.from_shape_poly(poly2d, "tyx", 0, 1)
    t0 = roi.t0

    ofs = ist_2d.estimate_shifts_absolute(t0, roi=roi)
    okay = (ofs - example.shifts + example.shifts[t0]) % (
        (1,) + img_2d.shape[-2:]
    )
    assert np.allclose(okay, 0), str(okay)


def test3d_roi():
    example = Example_TZYX()
    img_3d = example.create()
    ist_3d = ISTabilizer(img_3d, "tzyx")

    poly_3d = np.array(
        [
            [11.0, 7.0, 59.46602839, 35.94993883],
            [11.0, 7.0, 59.46602839, 66.41732531],
            [11.0, 7.0, 89.637615, 66.41732531],
            [11.0, 7.0, 89.637615, 35.94993883],
        ]
    )

    roi = ROIRect.from_shape_poly(poly_3d, "tzyx", 7, 15)
    t0 = roi.t0

    ofs = ist_3d.estimate_shifts_absolute(t0, roi=roi)
    okay = (ofs - example.shifts + example.shifts[t0]) % img_3d.shape[-3:]
    assert np.allclose(okay, 0), str(okay)


def test3d_C_roi():
    example = Example_TZYX()
    img_3d = example.create()

    img_3d = np.stack([img_3d, img_3d[::-1]], axis=1)

    ist_3d = ISTabilizer(img_3d, "tczyx")

    poly_3d = np.array(
        [
            [11.0, 0, 2.0, 59.46602839, 35.94993883],
            [11.0, 0, 7.0, 59.46602839, 66.41732531],
            [11.0, 0, 7.0, 89.637615, 66.41732531],
            [11.0, 0, 11.0, 89.637615, 35.94993883],
        ]
    )

    roi = ROIRect.from_shape_poly(poly_3d, "tzyx", 7, 15)
    t0 = roi.t0
    c0 = roi.c0
    ofs = ist_3d.estimate_shifts_absolute(t0, roi=roi)

    ofs = ist_3d.estimate_shifts_absolute(t0, roi=roi, channel=c0)
    okay = (ofs - example.shifts + example.shifts[t0]) % img_3d.shape[-3:]
    assert np.allclose(okay, 0), str(okay)
