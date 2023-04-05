"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy as np
from skimage.draw import disk
import tifffile


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    T = 32
    X = 256
    Y = 256

    img = np.zeros((T, Y, X), dtype="uint8")
    for t in range(T):
        img[t] += (np.random.rand(Y, X) * 64).astype("uint8")

    pos = np.array([32, 32], dtype="float64")

    t_ = 0
    for t in range(12):
        pos += 2 * t
        rr, cc = disk(pos, 8, shape=(256, 256))
        img[t_, rr, cc] = 255
        t_ += 1

    for t in range(10):
        pos += np.array([t * 1, -t * 2])
        rr, cc = disk(pos, 8, shape=(256, 256))
        img[t_, rr, cc] = 255
        t_ += 1

    for t in range(10):
        pos += np.array([-t * 2, -t * 1 / 2])
        rr, cc = disk(pos, 8, shape=(256, 256))
        img[t_, rr, cc] = 255
        t_ += 1

    return [(img, {})]
