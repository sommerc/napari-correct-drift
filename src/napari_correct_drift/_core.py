"""
doc
"""

import numpy as np
from scipy.ndimage import shift
from scipy.interpolate import interp1d
from skimage.registration import phase_cross_correlation

from napari.utils import progress


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

        self._check()

    def _check(self):
        assert self.z_max > self.z_min > -1, "z-dim mismatch"
        assert self.y_max > self.y_min > -1, "y-dim mismatch"
        assert self.x_max > self.x_min > -1, "x-dim mismatch"

    @classmethod
    def from_shape_poly(cls, shape_poly, dims, z_min, z_max):
        p_min = shape_poly.min(0)
        p_max = shape_poly.max(0)

        t = dims.find("t")
        t0 = int(p_min[t])

        x_min = int(p_min[-1] + 0.5)
        x_max = int(p_max[-1] + 0.5)

        y_min = int(p_min[-2] + 0.5)
        y_max = int(p_max[-2] + 0.5)

        if not "z" in dims:
            z_min = 0
            z_max = 1

        c0 = 0
        if "c" in dims:
            c = dims.find("c")
            c0 = int(p_min[c])

        return cls(x_min, x_max, y_min, y_max, z_min, z_max, t0, c0)

    @classmethod
    def from_bbox(cls, bbox, t0, c0):
        return cls(
            bbox[4], bbox[5], bbox[2], bbox[3], bbox[0], bbox[1], t0, c0
        )

    @property
    def origin(self):
        return self.bbox[::2]

    @property
    def shape(self):
        return self.bbox[1::2] - self.bbox[::2]

    @property
    def bbox(self):
        return np.array(
            [
                self.z_min,
                self.z_max,
                self.y_min,
                self.y_max,
                self.x_min,
                self.x_max,
            ]
        )

    def __str__(self):
        return f"""
        t0 {self.t0}
        c0 {self.c0}
        z: {self.z_min}, {self.z_max}
        y: {self.y_min}, {self.y_max}
        x: {self.x_min}, {self.x_max}
        
        bbox: {self.bbox}
        """


class ISTabilizer:
    def __init__(self, data, dims):
        assert (
            "t" in dims
        ), f"Axis 't' for stabilizing not found in data dims: '{dims}'"
        assert (
            "x" in dims and "y" in dims
        ), f"Axis x or y not present in dims: '{dims}'"

        self.is_multi_channel = "c" in dims
        self.is_3d = "z" in dims

        self.dims = dims
        self.data_arranger = ArrayRearranger("tczyx", dims)
        self.data = self.data_arranger(data)

        self.T, self.C, self.Z, self.Y, self.X = self.data.shape

    @staticmethod
    def iter_abs(T, t0, step):
        rm = np.c_[np.ones(T) * t0, np.arange(T)]
        rm_inc = rm[::step, :]

        # make sure last element is in iter (needed for interpolation)
        if not np.all(rm[-1] == rm_inc[-1]):
            rm = np.r_[rm_inc, rm[-1][None]]

        return rm.astype("int32")

    @staticmethod
    def iter_rel(T, t0, step):
        rm = []

        # forward
        r = t0
        while True:
            m = r + step
            if m < T:
                rm.append([r, m])
                r += step
            else:
                break
        # backward
        r = t0
        while True:
            m = r - step
            if m > -1:
                rm.append([r, m])
                r -= step
            else:
                break
        # make sure t0 is reference, if t0 < increment
        if 1 <= t0 < step:
            rm.append([t0, 0])

        return np.array(rm)

    def estimate_shifts_absolute(
        self,
        t0=0,
        channel=0,
        increment=1,
        upsample_factor=1,
        roi=None,
    ):
        if not self.is_multi_channel:
            channel = 0

        offsets = np.zeros((self.T, 3))
        offsets.fill(np.nan)

        ref_img = self.data[t0, channel]

        if roi is not None:
            ref_img_crop = ref_img[
                slice(max(0, roi.bbox[0]), min(ref_img.shape[0], roi.bbox[1])),
                slice(max(0, roi.bbox[2]), min(ref_img.shape[1], roi.bbox[3])),
                slice(max(0, roi.bbox[4]), min(ref_img.shape[2], roi.bbox[5])),
            ].copy()

            ref_img = np.zeros_like(ref_img)

            ref_img[
                : ref_img_crop.shape[0],
                : ref_img_crop.shape[1],
                : ref_img_crop.shape[2],
            ] = ref_img_crop

        for _, m in progress(self.iter_abs(self.T, t0, increment)):
            mov_img = self.data[m, channel]

            offset, _, _ = phase_cross_correlation(
                ref_img,
                mov_img,
                upsample_factor=upsample_factor,
                return_error="always",
                disambiguate=True,
            )

            offset = -np.asarray(offset, dtype="float32")

            if roi is not None:
                offset -= roi.bbox[::2]

            offsets[m] = offset

        return offsets

    def interpolate_offsets(self, offsets):
        x = np.nonzero(~np.isnan(offsets).any(axis=1))[0]
        m = np.nonzero(np.isnan(offsets).any(axis=1))[0]
        y = offsets[x, :]
        offsets[m] = interp1d(
            x, y, kind="linear", axis=0, fill_value="extrapolate"
        )(m)

        return offsets

    def apply_shifts(
        self,
        offsets,
        extend_output=False,
        order=1,
        mode="constant",
    ):
        if extend_output:
            # compute new shape of extended output
            shape_ext_zyx = np.ceil(offsets.max(0) - offsets.min(0)).astype(
                int
            )
            shape_yxz = np.array([self.Z, self.Y, self.X])
            out_shape_zyx = shape_ext_zyx + shape_yxz

            # create extended output
            output = np.zeros(
                (self.T, self.C) + tuple(out_shape_zyx), dtype="float32"
            )

            # split shifts into integer and subpixel parts.
            # the ndi.shift takes care of subpixel part
            # the interger shifts are handled via giving an output slice

            offsets_px = -np.ceil(offsets - offsets.max(0)).astype("int")
            offsets_sub = -(offsets - offsets.max(0)) - offsets_px

            for t in range(self.T):
                for c in range(self.C):
                    img = self.data[t, c]
                    output_view = output[
                        t,
                        c,
                        offsets_px[t, 0] : offsets_px[t, 0] + img.shape[0],
                        offsets_px[t, 1] : offsets_px[t, 1] + img.shape[1],
                        offsets_px[t, 2] : offsets_px[t, 2] + img.shape[2],
                    ]

                    shift(
                        img,
                        -offsets_sub[t],
                        output=output_view,
                        order=order,
                        mode=mode,
                        prefilter=False,
                    )

        else:
            output = np.zeros_like(self.data)
            for t in range(self.T):
                for c in range(self.C):
                    img = self.data[t, c]
                    output[t, c] = shift(
                        img,
                        -offsets[t],
                        order=order,
                        mode=mode,
                        prefilter=False,
                    )

        # Transfrom from TCZYX to original data layout
        return self.data_arranger.inv(output)

    def estimate_shifts_relative(
        self,
        t0=0,
        channel=0,
        increment=1,
        upsample_factor=1,
        roi=None,
    ):
        offsets_rel = np.zeros((self.T, 3))
        offsets_rel.fill(np.nan)

        if roi is not None:
            t0 = roi.t0
            c0 = roi.c0
            roi_t0 = ROIRect.from_bbox(roi.bbox, t0, c0)

        if not self.is_multi_channel:
            channel = 0

        for r, m in progress(self.iter_rel(self.T, t0, increment)):
            if (r == t0) and (roi is not None):
                mov_bbox = roi_t0.bbox.copy()

            ref_img = self.data[r, channel]
            mov_img = self.data[m, channel]

            if roi is not None:
                ref_img_crop = ref_img[
                    slice(
                        max(0, mov_bbox[0]), min(ref_img.shape[0], mov_bbox[1])
                    ),
                    slice(
                        max(0, mov_bbox[2]), min(ref_img.shape[1], mov_bbox[3])
                    ),
                    slice(
                        max(0, mov_bbox[4]), min(ref_img.shape[2], mov_bbox[5])
                    ),
                ].copy()

                ref_img = np.zeros_like(ref_img)

                ref_img[
                    : ref_img_crop.shape[0],
                    : ref_img_crop.shape[1],
                    : ref_img_crop.shape[2],
                ] = ref_img_crop

            offset, _, _ = phase_cross_correlation(
                ref_img,
                mov_img,
                upsample_factor=upsample_factor,
                return_error="always",
                disambiguate=True,
            )
            offset = np.asarray(offset, dtype="float32")

            if roi is not None:
                offset += mov_bbox[::2]

            if roi is not None:
                mov_bbox[::2] -= np.round(offset).astype("int32")
                mov_bbox[1::2] -= np.round(offset).astype("int32")

                mov_bbox[0:2] = np.clip(mov_bbox[0:2], 0, ref_img.shape[0])
                mov_bbox[2:4] = np.clip(mov_bbox[2:4], 0, ref_img.shape[1])
                mov_bbox[4:] = np.clip(mov_bbox[4:], 0, ref_img.shape[2])
            if m > r:
                offsets_rel[m] = -offset
            else:
                offsets_rel[r] = offset

        offsets_rel[0] = 0
        nan_row = np.nonzero(np.isnan(offsets_rel).any(axis=1))[0]
        offsets = np.nancumsum(offsets_rel, axis=0)
        offsets[nan_row, :] = np.nan

        offsets -= offsets[t0]

        return offsets
