"""Napari-Correct-Drift's core functionality classes

The actual cross-correlation is done with by Fourier phase correlation implmented in `phase_cross_correlation
<https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation>`_

If an ROI is used the the ROI-cropped images is zero-padded.
"""


import warnings
from functools import lru_cache
from typing import Callable, Tuple

import numpy as np
from napari.utils import progress
from scipy.interpolate import interp1d
from scipy.ndimage import shift
from scipy.signal import windows
from skimage.registration import phase_cross_correlation


@lru_cache
def window_nd(
    shape: Tuple[int, ...], win_func: Callable = windows.hann
) -> np.array:
    """N-dimensional windowing using 1d `win_func`. Default Hann window.

    Note 1: shape should be at least 2d

    Note 2: this function is cached.

    Args:
        shape (Tuple[int, ...]): Desired output shape
        win_func (Callable, optional): 1d Window function as in `scipy.signal.windows`. Defaults to `windows.hann`.

    Returns:
        np.array: Array with applied window function.
    """
    assert len(shape) > 1, "Shape must have minimum 2 elements"

    out = np.outer(win_func(shape[0]), win_func(shape[1]))

    for s in shape[2:]:
        out = np.outer(out, win_func(s))

    return out.reshape(shape)


class ArrayAxesStandardizer:
    """This class standardizes the axes of numpy arrays by rearranging
       the axes order.

       Axes orders are given by strings, e. g. 'tzcyx'

    Attributes:
        out_order (str): A string representing the desired output
            order of the array axes.
        in_order  (str): A string representing the current order of
            the array axes.

    """

    def __init__(self, out_order: str, in_order: str):
        """
        Initializes the class and checks for any invalid inputs.

        Args:
            out_order (str): A string representing the desired output
                order of the array axes.
            in_order (str): A string representing the current order of
                the array axes.

        Raises:
            AssertionError: If `in_order` contains any elements not in
                `out_order` or if `out_order` or `in_order` contains duplicates.
        """
        self._check_order_str(out_order)
        self._check_order_str(in_order)

        assert set(in_order).issubset(
            set(out_order)
        ), "in_order not subset of out_order"

        self.out_order = out_order
        self.in_order = in_order

    def _check_order_str(self, order: str):
        assert len(set(order)) == len(
            order
        ), f"Duplicates in order found: '{order}'"

    def __call__(self, data: np.array):
        """Standardizes given `data` array

        Takes an input numpy array and standardizes the axes according
        to the output order. It returns the standardized array.

        Args:
            data (np.array): A numpy array that needs to be standardized.

        Returns:
            np.array: A standardized numpy array.

        Raises:
            AssertionError: If the shape of the input array does not match the input order.
        """

        assert len(data.shape) == len(self.in_order), (
            f"Shape mismatch. Data shape is '{data.shape}'"
            ", but in order is '{self.in_order}'"
        )
        permute = []
        missing = []
        for i, dim in enumerate(self.out_order):
            j = self.in_order.find(dim)
            permute.append(j) if j > -1 else missing.append(i)

        data_rearranged = np.expand_dims(data.transpose(permute), missing)
        # assert not data_rearranged.flags["OWNDATA"], "not a view??"
        return data_rearranged

    def inv(self, data: np.array):
        """Inverse standardization

        Applies the inverse transform to the input data. The input data
        is expected to be a numpy array with the shape
        specified by the 'out_order' argument passed to the constructor.
        The output is a numpy array with the shape
        specified by the 'in_order' argument passed to the constructor.

        Args:
            data (np.array): The input data to be transformed.

        Returns:
            np.array: The transformed data.

        Raises:
            AssertionError: If the input data does not have the same shape
            as specified by the 'out_order' argument.
        """

        assert len(data.shape) == len(self.out_order), (
            f"Shape mismaatch. Data shape is '{data.shape}',"
            " but in order is '{self.out_order}'"
        )
        permute = []
        missing = []
        for i, dim in enumerate(self.out_order):
            j = self.in_order.find(dim)
            permute.append(j) if j > -1 else missing.append(i)

        res = np.transpose(
            np.squeeze(data, axis=tuple(missing)),
            self.inverse_permutation(permute),
        )
        return res

    @staticmethod
    def inverse_permutation(a):
        b = np.arange(len(a))
        b[a] = b.copy()
        return b


class ROIRect:
    """Helper classes for 3D bounding-box, localized in channels and time

    Attributes:
        x_min, x_max, y_min, y_max, z_min, z_max (int): special boundaries of the ROI rectangle.
        t0 (int): frame index
        c0 (int): channel index
    """

    def __init__(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        z_min: int,
        z_max: int,
        t0: int,
        c0: int,
    ):
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
        assert (
            self.z_max > self.z_min
        ), "Z-dim mismatch. Choose Z minimum and maximum"
        assert self.y_max > self.y_min, "y-dim mismatch"
        assert self.x_max > self.x_min, "x-dim mismatch"

    @classmethod
    def from_shape_poly(
        cls, shape_poly: np.array, dims: str, z_min: int, z_max: int
    ):
        """Creates ROIRect from napari shape polygon

        Args:
            shape_poly (np.array): napari polygon shape
            dims (str): dimensions string
            z_min (int): z min
            z_max (int): z max

        """
        p_min = shape_poly.min(0)
        p_max = shape_poly.max(0)

        t = dims.find("t")
        t0 = int(p_min[t])

        x_min = int(p_min[-1] + 0.5)
        x_max = int(p_max[-1] + 0.5)

        y_min = int(p_min[-2] + 0.5)
        y_max = int(p_max[-2] + 0.5)

        if "z" not in dims:
            z_min = 0
            z_max = 1

        c0 = 0
        if "c" in dims:
            c = dims.find("c")
            c0 = int(p_min[c])

        return cls(x_min, x_max, y_min, y_max, z_min, z_max, t0, c0)

    @classmethod
    def from_bbox(cls, bbox: np.array, t0: int, c0: int):
        """Creates ROIRect from  from bounding box list:

           [z_min, z_max, y_min, y_max, x_min, x_max,]

        Args:
            bbox (np.array): bounding box
            t0 (int): key frame
            c0 (int): key channel

        """
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


class CorrectDrift:
    """Main drift correction class

    Attributes:
        is_multi_channel (bool): True if the data has multiple channels.
        is_3d (bool): True if the data has a z dimension.
        dims (str): The dimension order of the input data.
        data_arranger (ArrayAxesStandardizer): An `ArrayAxesStandardizer` instance used to rearrange the input data.
        data (np.array): The rearranged input data.
        T, C, Z, Y, X (int): The number of time points, channels, z-slices, rows, and columns in the input data.
    """

    def __init__(self, data: np.array, dims: str):
        """Init of main drift correction class

        Args:
            data (np.array): image data
            dims (str): corresponding axes
        """
        assert (
            "t" in dims
        ), f"Axis 't' for stabilizing not found in data dims: '{dims}'"
        assert (
            "x" in dims and "y" in dims
        ), f"Axis x or y not present in dims: '{dims}'"

        self.is_multi_channel = "c" in dims
        self.is_3d = "z" in dims

        self.dims = dims
        self.data_arranger = ArrayAxesStandardizer("tczyx", dims)
        self.data = self.data_arranger(data)

        self.T, self.C, self.Z, self.Y, self.X = self.data.shape

    @staticmethod
    def iter_abs(T: int, t0: int, step: int) -> np.array:
        rm = np.c_[np.ones(T) * t0, np.arange(T)]
        rm_inc = rm[::step, :]

        # make sure last element is in iter (needed for interpolation)
        if not np.all(rm[-1] == rm_inc[-1]):
            rm = np.r_[rm_inc, rm[-1][None]]

        return rm.astype("int32")

    @staticmethod
    def iter_rel(T: int, t0: int, step: int) -> np.array:
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

    def estimate_drift(
        self,
        t0: int = 0,
        channel: int = 0,
        increment: int = 1,
        upsample_factor: int = 1,
        roi: ROIRect = None,
        normalization: str = "phase",
        mode: str = "relative",
        max_shifts: Tuple[int, int, int] = None,
        use_window: bool = True,
    ):
        """Estimate drift entry point.

        Calls `_estimate_drift_relative` or `_estimate_drift_absolute` depending on given `mode`

        Args:
            t0 (int, optional): key frame. Defaults to 0.
            channel (int, optional): key channel. Defaults to 0.
            increment (int, optional): frame increment. Defaults to 1.
            upsample_factor (int, optional): upsample factor. Defaults to 1.
            roi (ROIRect, optional): ROI. Defaults to None.
            normalization (str, optional): normalization. Defaults to "phase".
            mode (str, optional): mode of drift correction. Defaults to "relative".
            max_shifts (Tuple[int, int, int], optional): maximum allowed shifts. Defaults to None.
            use_winsow (bool): use Hann window to prevent spectral leakage. Defautls to True.

        Raises:
            AttributeError: if `mode` is not supported

        """
        if mode == "relative":
            return self._estimate_drift_relative(
                t0=t0,
                channel=channel,
                increment=increment,
                upsample_factor=upsample_factor,
                normalization=normalization,
                roi=roi,
                max_shifts=max_shifts,
                windowing=use_window,
            )

        elif mode == "absolute":
            return self._estimate_drift_absolute(
                t0=t0,
                channel=channel,
                increment=increment,
                upsample_factor=upsample_factor,
                normalization=normalization,
                roi=roi,
                windowing=use_window,
            )

        else:
            raise AttributeError(f"Estimation mode '{mode}' not valid")

    def _estimate_drift_relative(
        self,
        t0: int = 0,
        channel: int = 0,
        increment: int = 1,
        upsample_factor: int = 1,
        normalization: str = "phase",
        roi: ROIRect = None,
        max_shifts: tuple = None,
        windowing: bool = True,
    ):
        offsets_rel = np.zeros((self.T, 3))
        offsets_rel.fill(np.nan)

        if max_shifts is None:
            max_shifts = [np.Inf] * 3

        if roi is not None:
            t0 = roi.t0
            c0 = roi.c0
            roi_t0 = ROIRect.from_bbox(roi.bbox, t0, c0)

        if not self.is_multi_channel:
            channel = 0

        for r, m in progress(
            self.iter_rel(self.T, t0, increment),
            desc=f"Estimate drift (relative to frame '{t0}')",
        ):
            if (r == t0) and (roi is not None):
                # reset ROI, now it goes downward
                mov_bbox = roi_t0.bbox.copy()

            ref_img = self.data[r, channel]
            mov_img = self.data[m, channel]

            if windowing:
                mov_img = mov_img * window_nd(mov_img.shape)

            if roi is None:
                if windowing:
                    ref_img = ref_img * window_nd(ref_img.shape)
            else:
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

                # Windowing with hann window
                if windowing:
                    ref_img_crop = ref_img_crop * window_nd(ref_img_crop.shape)

                ref_img = np.zeros_like(ref_img)

                ref_img[
                    : ref_img_crop.shape[0],
                    : ref_img_crop.shape[1],
                    : ref_img_crop.shape[2],
                ] = ref_img_crop

            # see https://github.com/scikit-image/scikit-image/issues/7089
            disambiguate = True
            if mov_img.std() == 0:
                disambiguate = False

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    category=RuntimeWarning,
                )

                offset_raw, _, _ = phase_cross_correlation(
                    ref_img,
                    mov_img,
                    upsample_factor=upsample_factor,
                    disambiguate=disambiguate,
                    normalization=normalization,
                )

            offset_raw = np.asarray(offset_raw, dtype="float32")

            if roi is None:
                # no roi offsets stay unchanged
                offset = offset_raw
            else:
                # offset is relative to ROI content pasted at origin
                # hence, add roi origin
                offset = offset_raw + mov_bbox[::2]

                # prepare mov_bbox for next round and move roi along
                mov_bbox[::2] -= np.round(offset).astype("int32")
                mov_bbox[1::2] -= np.round(offset).astype("int32")

                # clip if new moving bbox is out of image
                mov_bbox[0:2] = np.clip(mov_bbox[0:2], 0, ref_img.shape[0])
                mov_bbox[2:4] = np.clip(mov_bbox[2:4], 0, ref_img.shape[1])
                mov_bbox[4:] = np.clip(mov_bbox[4:], 0, ref_img.shape[2])

            # set offsets depending on direction relative to key-frame
            if m > r:
                if not np.any(np.abs(offset) > np.array(max_shifts)):
                    offsets_rel[m] = -offset
            else:
                if not np.any(np.abs(offset) > np.array(max_shifts)):
                    offsets_rel[r] = offset

        offsets_rel[0] = 0
        nan_row = np.nonzero(np.isnan(offsets_rel).any(axis=1))[0]
        offsets = np.nancumsum(offsets_rel, axis=0)
        offsets[nan_row, :] = np.nan

        offsets -= offsets[t0]

        return offsets

    def _estimate_drift_absolute(
        self,
        t0: int = 0,
        channel: int = 0,
        increment: int = 1,
        upsample_factor: int = 1,
        normalization: str = "phase",
        roi: ROIRect = None,
        windowing: bool = True,
    ):
        if not self.is_multi_channel:
            channel = 0

        offsets = np.zeros((self.T, 3))
        offsets.fill(np.nan)

        ref_img = self.data[t0, channel]

        if roi is None:
            if windowing:
                ref_img = ref_img * window_nd(ref_img.shape)
        else:
            ref_img_crop = ref_img[
                slice(max(0, roi.bbox[0]), min(ref_img.shape[0], roi.bbox[1])),
                slice(max(0, roi.bbox[2]), min(ref_img.shape[1], roi.bbox[3])),
                slice(max(0, roi.bbox[4]), min(ref_img.shape[2], roi.bbox[5])),
            ].copy()

            if windowing:
                ref_img_crop = ref_img_crop * window_nd(ref_img_crop.shape)

            ref_img = np.zeros_like(ref_img)

            ref_img[
                : ref_img_crop.shape[0],
                : ref_img_crop.shape[1],
                : ref_img_crop.shape[2],
            ] = ref_img_crop

        for _, m in progress(
            self.iter_abs(self.T, t0, increment),
            desc=f"Estimate drift (absolute to frame '{t0}')",
        ):
            mov_img = self.data[m, channel]

            # see https://github.com/scikit-image/scikit-image/issues/7089
            disambiguate = True
            if mov_img.std() == 0:
                disambiguate = False

            if windowing:
                mov_img = mov_img * window_nd(mov_img.shape)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    category=RuntimeWarning,
                )
                offset, _, _ = phase_cross_correlation(
                    ref_img,
                    mov_img,
                    upsample_factor=upsample_factor,
                    disambiguate=disambiguate,
                    normalization=normalization,
                )

            offset = -np.asarray(offset, dtype="float32")

            if roi is not None:
                offset -= roi.bbox[::2]

            offsets[m] = offset

        return offsets

    def interpolate_drift(self, offsets: np.array):
        """Linear interpolation of missing values given as `np.nan` values

        Args:
            offsets (np.array): The drift table containing missing values (`np.nan`)

        Raises:
            RuntimeError: When too few values for interpolation

        Returns:
            offsets (np.array): offsets with linearly interpolated values
        """
        x = np.nonzero(~np.isnan(offsets).any(axis=1))[0]

        if x.shape[0] < 3:
            raise RuntimeError(
                "Too little shift values to interpolate missing shifts. Disable outlier handling and/or use lower increment."
            )

        m = np.nonzero(np.isnan(offsets).any(axis=1))[0]
        y = offsets[x, :]
        offsets[m] = interp1d(
            x, y, kind="linear", axis=0, fill_value="extrapolate"
        )(m)

        return offsets

    def apply_drifts(
        self,
        offsets: np.array,
        extend_output: bool = False,
        order: int = 1,
        mode: str = "constant",
    ):
        """Apply drifts to effectively create the corrected output image

        Args:
            offsets (np.array): the drift table
            extend_output (bool, optional): if True, the output array is
                extended to always contain the original frame. Defaults to False.
            order (int, optional): order of interpolation (1=linear). Defaults to 1.
            mode (str, optional): border treatment. Defaults to "constant".

        Returns:
            np.array: the corrected output image
        """
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
            # the integer shifts are handled via giving an output slice

            offsets_px = -np.ceil(offsets - offsets.max(0)).astype("int")
            offsets_sub = -(offsets - offsets.max(0)) - offsets_px

            for t in progress(range(self.T), desc="Applying drift"):
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
                        offsets_sub[t],
                        output=output_view,
                        order=order,
                        mode=mode,
                        prefilter=False,
                    )

        else:
            output = np.zeros_like(self.data)
            for t in progress(range(self.T), desc="Applying drift"):
                for c in range(self.C):
                    img = self.data[t, c]
                    output[t, c] = shift(
                        img,
                        -offsets[t],
                        order=order,
                        mode=mode,
                        prefilter=False,
                    )

        # Transform from TCZYX to original data layout
        return self.data_arranger.inv(output)
