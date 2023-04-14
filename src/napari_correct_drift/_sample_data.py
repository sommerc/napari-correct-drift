"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy as np
from skimage.draw import disk
from skimage.morphology import ball, binary_dilation


class Example_TZYX:
    def __init__(self):
        deltas_3d = (
            [[1, 6, 4]] * 4
            + [[-1, 4, 1]] * 6
            + [[1, 2, -3]] * 6
            + [[0, -2, 7]] * 4
            + [[0, 2, -5]] * 4
            + [[0, 7, 0]] * 4
            + [[0, 0, -7]] * 3
        )

        shifts_3d = np.cumsum(deltas_3d, 0)
        shifts_3d = np.vstack(
            [
                [0] * 3,
                shifts_3d,
            ]
        )

        self.shape = (32, 16, 128, 80)
        self.start_pos = (8, 24, 32)
        self.deltas = deltas_3d
        self.shifts = shifts_3d

    def create(self, noise_level=0):
        deltas = np.asarray(self.deltas)

        T, Z, Y, X = self.shape
        img = np.zeros((T, Z, Y, X), dtype="bool")

        t = 0
        img[t][self.start_pos] = 1
        img[t] = binary_dilation(img[t], ball(3))

        pos = np.array(self.start_pos, dtype="int32")

        for d in deltas:
            t += 1
            pos += d
            img[t][tuple(pos)] = 1
            img[t] = binary_dilation(img[t], ball(3))

        img = img.astype("uint8") * (255 - noise_level)

        for t in range(T):
            img[t] += (np.random.rand(Y, X) * noise_level).astype("uint8")

        return img


class Example_TYX:
    def __init__(self):
        deltas_2d = [[8, 8]] * 10
        deltas_2d += [[-2, 6]] * 5
        deltas_2d += [[6, 3]] * 10
        deltas_2d += [[0, 7]] * 3
        deltas_2d += [[11, -3]] * 3

        deltas_2d = [(0, y, x) for y, x in deltas_2d]

        shifts_2d = np.cumsum(deltas_2d, 0)
        shifts_2d = np.vstack([[0, 0, 0], shifts_2d])

        self.shape = (32, 256, 256)
        self.start_pos = (32, 32)
        self.deltas = deltas_2d
        self.shifts = shifts_2d

    def create(self, noise_level=0):
        deltas = np.asarray(self.deltas)

        T, Y, X = self.shape
        img = np.zeros((T, Y, X), dtype="uint8")

        for t in range(T):
            img[t] += (np.random.rand(Y, X) * noise_level).astype("uint8")

        t = 0
        rr, cc = disk(self.start_pos, 8, shape=(256, 256))
        img[t, rr, cc] = 255

        pos = np.array(self.start_pos)

        for d in deltas:
            t += 1
            pos += d[1:]
            rr, cc = disk(pos, 8, shape=(256, 256))
            img[t, rr, cc] = 255

        return img


class Example_TCYX(Example_TYX):
    def create(self, noise_level=0):
        img = super().create(noise_level=noise_level)

        img = np.stack([img, img[::-1], np.zeros_like(img)], axis=1)
        self.shape = img.shape

        return img


def sample_2d():
    return [
        (
            Example_TYX().create(noise_level=16),
            {"name": "test_TYX", "contrast_limits": (0, 64)},
        ),
    ]


def sample_3d():
    return [
        (
            Example_TZYX().create(noise_level=16),
            {"name": "test_TZYX", "contrast_limits": (0, 64)},
        ),
    ]


def sample_3d_ch():
    return [
        (
            Example_TCYX().create(noise_level=16),
            {
                "name": "test_TCYX",
                "contrast_limits": (0, 64),
            },
        ),
    ]
