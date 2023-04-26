import numpy as np

from napari_correct_drift._core import CorrectDrift, ROIRect
from napari_correct_drift._sample_data import Example_TYX, Example_TZYX


def test_tyx():
    example = Example_TYX()
    img_2d = example.create()
    ist_2d = CorrectDrift(img_2d, "tyx")

    for t0 in [0, 17]:
        ofs = ist_2d._estimate_drift_absolute(t0)
        ok = ofs - example.shifts + example.shifts[t0]
        assert np.allclose(ok, 0), "Shift estimation mismatch. Diff:\n"


def test_tzyx():
    example = Example_TZYX()
    img_3d = example.create()
    ist_3d = CorrectDrift(img_3d, "tzyx")

    for t0 in [0, 16]:
        ofs = ist_3d._estimate_drift_absolute(t0)
        ok = ofs - example.shifts + example.shifts[t0]
        assert np.allclose(ok, 0), "Shift estimation mismatch. Diff:\n"


def test_tcyx():
    example = Example_TYX()
    img_2d = example.create()
    img_2d = np.stack([img_2d, img_2d // 2], axis=1)
    ist_2d = CorrectDrift(img_2d, "tcyx")

    for t0 in [0, 7]:
        ofs = ist_2d._estimate_drift_absolute(t0, channel=0)
        ok = ofs - example.shifts + example.shifts[t0]
        assert np.allclose(ok, 0), "Shift estimation mismatch. Diff:\n"


def test_tczyx():
    example = Example_TZYX()
    img_3d = example.create()

    img_3d = np.stack([img_3d, img_3d // 2], axis=1)
    ist_3d = CorrectDrift(img_3d, "tczyx")

    for t0 in [0, 16]:
        ofs = ist_3d._estimate_drift_absolute(t0, channel=0)
        ok = ofs - example.shifts + example.shifts[t0]
        assert np.allclose(ok, 0), "Shift estimation mismatch."


def test_tyx_roi():
    example = Example_TYX()
    img_2d = example.create()
    ist_2d = CorrectDrift(img_2d, "tyx")

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

    ofs = ist_2d._estimate_drift_absolute(t0, roi=roi)
    ok = ofs - example.shifts + example.shifts[t0]
    assert np.allclose(ok, 0), "Shift estimation mismatch."


def test_tzyx_roi():
    example = Example_TZYX()
    img_3d = example.create()
    ist_3d = CorrectDrift(img_3d, "tzyx")

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

    ofs = ist_3d._estimate_drift_absolute(t0, roi=roi)
    ok = ofs - example.shifts + example.shifts[t0]
    assert np.allclose(ok, 0), "Shift estimation mismatch."


def test_tczyx_roi():
    example = Example_TZYX()
    img_3d = example.create()

    img_3d = np.stack([img_3d, img_3d[::-1]], axis=1)

    ist_3d = CorrectDrift(img_3d, "tczyx")

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
    ofs = ist_3d._estimate_drift_absolute(t0, roi=roi)

    ofs = ist_3d._estimate_drift_absolute(t0, roi=roi, channel=c0)
    ok = ofs - example.shifts + example.shifts[t0]
    assert np.allclose(ok, 0)


def test_tyx_rel():
    example = Example_TYX()
    img_2d = example.create()
    ist_2d = CorrectDrift(img_2d, "tyx")

    for t0 in [0, 6, 13]:
        ofs = ist_2d._estimate_drift_relative(t0)
        ok = ofs - example.shifts + example.shifts[t0]
        assert np.allclose(ok, 0)


def test_tczyx_rel():
    example = Example_TZYX()
    img_3d = example.create()

    img_3d = np.stack([img_3d, img_3d[:, ::-1] // 2], axis=1)

    ist_3d = CorrectDrift(img_3d, "tczyx")

    for t0 in [0, 16]:
        ofs = ist_3d._estimate_drift_relative(t0, channel=0)
        ok = ofs - example.shifts + example.shifts[t0]
        assert np.allclose(ok, 0), str(ok)


def test_tyx_roi_rel():
    example = Example_TYX()
    img_2d = example.create()
    ist_2d = CorrectDrift(img_2d, "tyx")

    poly2d = np.array(
        [
            [23.0, 122.57642757, 142.27071085],
            [23.0, 122.57642757, 190.77701484],
            [23.0, 181.65929409, 190.77701484],
            [23.0, 181.65929409, 142.27071085],
        ]
    )

    start_roi = ROIRect.from_shape_poly(poly2d, "tyx", 0, 1)
    t0 = start_roi.t0

    ofs = ist_2d._estimate_drift_relative(t0, roi=start_roi)
    ok = ofs - example.shifts + example.shifts[t0]
    assert np.allclose(ok, 0), str(ok)


def test_zcyx_roi_rel():
    example = Example_TZYX()
    img_3d = example.create()

    img_3d = np.stack([img_3d, img_3d[::-1]], axis=1)
    ist_3d = CorrectDrift(img_3d, "tczyx")

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
    assert t0 == 11
    assert c0 == 0
    assert roi.z_min == 7
    assert roi.z_max == 15

    ofs = ist_3d._estimate_drift_relative(t0, roi=roi)

    ofs = ist_3d._estimate_drift_relative(t0, roi=roi, channel=c0)
    ok = ofs - example.shifts + example.shifts[t0]

    assert np.allclose(ok, 0), str(ok)


def test_tyx_big_roi_rel():
    example = Example_TYX()
    img_2d = example.create()
    ist_2d = CorrectDrift(img_2d, "tyx")

    poly2d = np.array(
        [
            [23.0, 2.57642757, 4.27071085],
            [23.0, 2.57642757, 250.77701484],
            [23.0, 250.65929409, 250.77701484],
            [23.0, 250.65929409, 4.27071085],
        ]
    )

    start_roi = ROIRect.from_shape_poly(poly2d, "tyx", 0, 1)
    t0 = start_roi.t0

    ofs = ist_2d._estimate_drift_relative(t0, roi=start_roi)

    ok = ofs - example.shifts + example.shifts[t0]
    assert np.allclose(ok, 0)


def test_roirect():
    poly_3d = np.array(
        [
            [11.0, 3.0, 11.0, 59.46602839, 35.94993883],
            [11.0, 3.0, 11.0, 59.46602839, 66.41732531],
            [11.0, 3.0, 11.0, 89.637615, 66.41732531],
            [11.0, 3.0, 11.0, 89.637615, 35.94993883],
        ]
    )

    roi = ROIRect.from_shape_poly(poly_3d, "tczyx", 7, 15)
    t0 = roi.t0
    c0 = roi.c0
    assert t0 == 11
    assert c0 == 3
    assert roi.z_min == 7
    assert roi.z_max == 15
    assert roi.y_max == 90
    assert roi.x_min == 36

    roi_clone = ROIRect.from_bbox(roi.bbox, t0, c0)
    assert np.allclose(roi.bbox, roi_clone.bbox)
    assert np.allclose(roi.c0, roi_clone.c0)
