"""
doc
"""

from itertools import tee, chain
from functools import partialmethod

import numpy as np
from scipy.ndimage import shift
from skimage.draw import polygon2mask
from scipy.interpolate import interp1d
from skimage.registration import phase_cross_correlation

from scipy import fft


def _upsampled_dft(
    data, upsampled_region_size, upsample_factor=1, axis_offsets=None
):
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [
            upsampled_region_size,
        ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError(
                "shape of upsampled region sizes must be equal "
                "to input data's number of dimensions."
            )

    if axis_offsets is None:
        axis_offsets = [
            0,
        ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError(
                "number of axis offsets must be equal to input "
                "data's number of dimensions."
            )

    im2pi = 1j * 2 * np.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * fft.fftfreq(
            n_items, upsample_factor
        )
        kernel = np.exp(-im2pi * kernel)
        # use kernel with same precision as the data
        kernel = kernel.astype(data.dtype, copy=False)

        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def phase_cross_correlation2(
    reference_image, moving_image, *, upsample_factor=1, normalization="phase"
):
    src_freq = fft.fftn(reference_image)
    target_freq = fft.fftn(moving_image)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == "phase":
        eps = np.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError("normalization must be either phase or None")
    cross_correlation = fft.ifftn(image_product)

    # Locate maximum
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    float_dtype = image_product.real.dtype

    shifts = np.stack(maxima).astype(float_dtype, copy=False)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        pass
    else:
        # Initial shift estimate in upsampled grid
        upsample_factor = np.array(upsample_factor, dtype=float_dtype)
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        ).conj()
        # Locate maximum and map back to original pixel grid
        maxima = np.unravel_index(
            np.argmax(np.abs(cross_correlation)), cross_correlation.shape
        )
        CCmax = cross_correlation[maxima]

        maxima = np.stack(maxima).astype(float_dtype, copy=False)
        maxima -= dftshift

        shifts += maxima / upsample_factor

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts


class ArrayRearranger:
    """
    Example:

    img = np.zeros((8,3,64,128))
    in_order  = "zcxy"
    out_order = "tczxy"
    ara = ArrayRearranger(out_order, in_order)
    print("initial shape     :", img.shape, in_order)
    print("arranged in shape :", ara(img).shape, out_order)
    print("inverted shape    :", ara.inv(a(img)).shape, in_order)
    """

    def __init__(self, out_order, in_order):
        self._check_order_str(out_order)
        self._check_order_str(in_order)

        assert set(in_order).issubset(
            set(out_order)
        ), "in_order not subset of out_order"

        self.out_order = out_order
        self.in_order = in_order

    def _check_order_str(self, order):
        assert len(set(order)) == len(
            order
        ), f"Duplicates in order found: '{order}'"

    def __call__(self, data):
        assert len(data.shape) == len(
            self.in_order
        ), f"Shape mismatch. Data shape is '{data.shape}', but in order is '{self.in_order}'"
        permute = []
        missing = []
        for i, dim in enumerate(self.out_order):
            j = self.in_order.find(dim)
            permute.append(j) if j > -1 else missing.append(i)

        data_rearranged = np.expand_dims(data.transpose(permute), missing)
        assert not data_rearranged.flags["OWNDATA"], "not a view??"
        return data_rearranged

    def inv(self, data):
        assert len(data.shape) == len(
            self.out_order
        ), f"Shape mismaatch. Data shape is '{data.shape}', but in order is '{self.out_order}'"
        permute = []
        missing = []
        for i, dim in enumerate(self.out_order):
            j = self.in_order.find(dim)
            permute.append(j) if j > -1 else missing.append(i)

        # print(data.shape, permute, missing)
        res = np.transpose(
            np.squeeze(data, axis=tuple(missing)),
            self.inverse_permutation(permute),
        )
        assert not res.flags["OWNDATA"], "not a view??"
        return res

    @staticmethod
    def inverse_permutation(a):
        b = np.arange(len(a))
        b[a] = b.copy()
        return b


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def fast_int_shift(array, shift, cval=0):
    assert array.ndim == len(
        shift
    ), f"Array dimensions do not match shifts array.ndim='{array.ndim}' with shifts='{shift}'"

    # shift = np.array(shift).astype("int32")

    res = np.roll(array, shift, axis=tuple(range(array.ndim)))
    for d, s in enumerate(shift):
        if s > 0:
            sl = [slice(None)] * len(shift)
            sl[d] = slice(0, s)
            res[tuple(sl)] = cval
        if s < 0:
            sl = [slice(None)] * len(shift)
            sl[d] = slice(s, None)
            res[tuple(sl)] = cval

    return res


class ROIRect:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, t0, c0):
        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self.z_min = z_min
        self.z_max = z_max

        self.t0 = t0
        self.c0 = c0

        self.origin = np.array([z_min, y_min, x_min])

    @classmethod
    def from_shape_poly(cls, shape_poly, dims, z_min, z_max):
        p_min = shape_poly.min(0)
        p_max = shape_poly.max(0)

        t = dims.find("t")
        t0 = int(p_min[t])

        x_min = int(p_min[-1])
        x_max = int(p_max[-1])

        y_min = int(p_min[-2])
        y_max = int(p_max[-2])

        if not "z" in dims:
            z_min = 0
            z_max = 1

        c0 = 0
        if "c" in dims:
            c = dims.find("c")
            c0 = int(p_min[c])

        return cls(x_min, x_max, y_min, y_max, z_min, z_max, t0, c0)


class ISTabilizer:
    def __init__(self, data, dims):
        assert (
            "t" in dims
        ), f"Axis 't' for stabilizing not found in data dims: '{dims}'"
        assert (
            "x" in dims and "y" in dims
        ), f"Axis x or y not present in dims: '{dims}'"

        self.multi_channel = "c" in dims
        self.is_3d = "z" in dims

        self.dims = dims
        self.data_arranger = ArrayRearranger("tczyx", dims)
        self.data = self.data_arranger(data)

        self.T, self.C, self.Z, self.Y, self.X = self.data.shape

    def set_roi(self, roi):
        self.roi = roi

    def crop_pad_roi(self, ref_img):
        ref_img_crop = ref_img[
            slice(self.roi.z_min, self.roi.z_max),
            slice(self.roi.y_min, self.roi.y_max),
            slice(self.roi.x_min, self.roi.x_max),
        ]

        diffs_back = np.array(ref_img.shape) - np.array(ref_img_crop.shape)
        diffs_front = np.zeros_like(diffs_back)

        REF_MEAN = ref_img.mean()

        return np.pad(
            ref_img_crop,
            list(zip(diffs_front, diffs_back)),
            mode="constant",
            constant_values=REF_MEAN,
        )

    def estimate_shifts_relative(
        self,
        t0,
        channel=0,
        increment=1,
        upsample_factor=1,
        roi_mask=None,
    ):
        if not self.multi_channel:
            channel = 0

        t_range_forward = list(range(t0, self.T, increment))
        if len(t_range_forward) > 0 and t_range_forward[-1] != self.T - 1:
            t_range_forward.append(self.T - 1)

        t_range_backward = list(range(t0, 0, -increment))
        if len(t_range_backward) > 0 and t_range_backward[-1] != 0:
            t_range_backward.append(0)

        offsets = np.zeros((self.T, 3 if self.is_3d else 2))
        offsets.fill(np.nan)
        if roi_mask is not None:
            offsets[t0] = roi_mask.bbox[[0, 2]]

        cnt = 0
        for r, m in chain(
            pairwise(t_range_forward), pairwise(t_range_backward)
        ):
            if self.is_3d:
                ref_img = self.data[r, channel]
                mov_img = self.data[m, channel]
            else:
                ref_img = self.data[r, channel, 0]
                mov_img = self.data[m, channel, 0]

            if roi_mask is not None:
                ref_img_crop = ref_img[
                    ...,
                    slice(
                        int(offsets[r][0]),
                        int(offsets[r][0] + roi_mask.height),
                    ),
                    slice(
                        int(offsets[r][1]),
                        int(offsets[r][1] + roi_mask.width),
                    ),
                ]

                # if cnt == 0:
                diffs = np.array(mov_img.shape) - np.array(ref_img_crop.shape)

                diffs_a = np.zeros_like(diffs)
                diffs_b = diffs
                REF_MEAN = ref_img.mean()

                ref_img = np.pad(
                    ref_img_crop,
                    list(zip(diffs_a, diffs_b)),
                    mode="constant",
                    constant_values=REF_MEAN,
                )

            offset = phase_cross_correlation(
                ref_img,
                mov_img,
                upsample_factor=upsample_factor,
                return_error=False,
            )

            offset = -offset

            print(" - r, m:", r, m, offset)
            offsets[m] = -offset

            cnt += 1
            if cnt > 25:
                pass
                # break

        if increment > 1:
            x = np.nonzero(~np.isnan(offsets).any(axis=1))[0]
            m = np.nonzero(np.isnan(offsets).any(axis=1))[0]
            y = offsets[x, :]
            offsets[m] = interp1d(x, y, kind="linear", axis=0)(m)

        offsets -= offsets[t0]
        return offsets

    def estimate_shifts_absolute(
        self,
        t0,
        channel=0,
        increment=1,
        upsample_factor=1,
        roi=None,
    ):
        if not self.multi_channel:
            channel = 0

        def iter_abs(T, t0, step):
            a = np.c_[np.ones(T) * t0, np.arange(T)]
            b = a[::step]
            if not np.all(b[-1] == a[-1]):
                b = np.r_[b, a[-1][None]]

            return b.astype("int32")

        offsets = np.zeros((self.T, 3))
        offsets.fill(np.nan)

        ref_img = self.data[t0, channel]

        if roi is not None:
            self.set_roi(roi)
            ref_img = self.crop_pad_roi(ref_img)

        for r, m in iter_abs(self.T, t0, increment):
            mov_img = self.data[m, channel]

            offset, _, _ = phase_cross_correlation(
                ref_img,
                mov_img,
                upsample_factor=1.0,
                return_error="always",
                disambiguate=True,
                overlap_ratio=0.1,
            )

            offset = -np.asarray(offset, dtype="float32")

            if roi is not None:
                offset -= self.roi.origin

            offsets[m] = offset

        if increment > 1:
            x = np.nonzero(~np.isnan(offsets).any(axis=1))[0]
            m = np.nonzero(np.isnan(offsets).any(axis=1))[0]
            y = offsets[x, :]
            offsets[m] = interp1d(x, y, kind="linear", axis=0)(m)

        return offsets

    def apply_shifts(
        self,
        offsets,
        use_3d=True,
        interpolation_order=1,
        boundary_mode="constant",
    ):
        output = np.zeros_like(self.data)

        # if self.is_3d and not use_3d:
        #     offsets[0, :] = 0

        for t in range(self.T):
            for c in range(self.C):
                img = self.data[t, c]
                output[t, c] = shift(
                    img,
                    -offsets[t],
                    order=interpolation_order,
                    mode=boundary_mode,
                    prefilter=False,
                )

        return self.data_arranger.inv(output)
