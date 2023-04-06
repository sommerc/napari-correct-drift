"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from numpy.lib.arraysetops import isin

from ._core import ISTabilizer, ROIRect

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QWidget,
    QPushButton,
    QSlider,
    QCheckBox,
    QLabel,
    QSpinBox,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QComboBox,
    QGridLayout,
    QGroupBox,
)

from napari.layers.image.image import Image as IMAGE_LAYER
from napari.layers.shapes.shapes import Shapes as SHAPE_LAYER

if TYPE_CHECKING:
    import napari


def istabilize(image, dims, **kwargs):
    ist = ISTabilizer(
        image,
        dims,
    )

    if kwargs["stabilize_mode"] == "absolute":
        esitmate_offsets_func = ist.estimate_shifts_absolute

    else:
        esitmate_offsets_func = ist.estimate_shifts_relative

    offsets = esitmate_offsets_func(
        t0=kwargs["t0"],
        channel=kwargs["channel"],
        increment=kwargs["increment"],
        upsample_factor=kwargs["upsample_factor"],
        roi=kwargs["roi"],
    )

    res = ist.apply_shifts(offsets, use_3d=kwargs["use_3d"])
    return res


class CorrectDriftDock(QWidget):
    def _init_run(self):
        ### Input Image
        self.input_layer = QComboBox()
        self.input_image_layers = []

        def update_input_layer(event=None):
            print(event)
            self.input_layer.clear()
            self.input_image_layers = []
            for layer in self.viewer.layers:
                if isinstance(layer, IMAGE_LAYER):
                    self.input_image_layers.append(layer)
                    self.input_layer.addItem(layer.name)

        update_input_layer()
        print(self.input_image_layers)

        self.viewer.layers.events.inserted.connect(update_input_layer)
        self.viewer.layers.events.removed.connect(update_input_layer)

        # run button
        self.run_button = QPushButton("Correct Drift...")

        # add them
        run_panel = QGroupBox("Correct Drift")
        run_layout = QHBoxLayout()
        run_layout.addWidget(self.input_layer)
        run_layout.addWidget(self.run_button)

        run_panel.setLayout(run_layout)
        self.main_layout.addWidget(run_panel)

    def _init_axis_selection(self):
        axis_panel = QGroupBox("Define Axes")
        axis_layout = QGridLayout()

        axis_panel.setLayout(axis_layout)
        self.main_layout.addWidget(axis_panel)

        def update_axes_selection(event=None):
            layer = self.input_image_layers[self.input_layer.currentIndex()]

            *extra_dims, _, _ = layer.data.shape

            self.axis_combos = {}
            for d, d_size in enumerate(extra_dims):
                tmp = QComboBox()
                tmp.addItems(
                    [
                        f"Time ({d_size})",
                        f"Channels ({d_size})",
                        f"Z-axis ({d_size})",
                    ]
                )
                tmp.setCurrentIndex(d)
                self.axis_combos[d] = tmp

            for i in reversed(range(axis_layout.count())):
                axis_layout.itemAt(i).widget().setParent(None)

            for d, combo in self.axis_combos.items():
                axis_layout.addWidget(QLabel(f"Axis {d}: "), d + 1, 0)
                axis_layout.addWidget(combo, d + 1, 1)

        self.input_layer.currentIndexChanged.connect(update_axes_selection)

    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()

        from napari.qt.threading import (
            thread_worker,
            create_worker,
        )  # delayed import

        self.main_layout = QVBoxLayout()

        self._init_run()

        self._init_axis_selection()

        ### Axis selecton

        # use roi
        self.roi_checkbox = QCheckBox()
        self.roi_checkbox.setChecked(False)

        def add_shapes_layer(checked):
            if checked:
                for layer in self.viewer.layers:
                    if isinstance(layer, SHAPE_LAYER):
                        break
                else:
                    ndim = self.input_image_layers[
                        self.input_layer.currentIndex()
                    ].ndim
                    print(ndim)
                    self.viewer.add_shapes(ndim=ndim)

        self.roi_checkbox.toggled.connect(add_shapes_layer)

        self.roi_z_min_spin = QSpinBox()
        self.roi_z_min_spin.setMinimum(0)
        self.roi_z_min_spin.setValue(0)

        self.roi_z_max_spin = QSpinBox()
        self.roi_z_max_spin.setMinimum(-1)
        self.roi_z_max_spin.setValue(-1)

        # use 3D
        self.use_3d_checkbox = QCheckBox()
        self.use_3d_checkbox.setChecked(False)

        # increment
        self.increment_box = QSpinBox()
        self.increment_box.setMinimum(1)
        self.increment_box.setMaximum(16)
        self.increment_box.setValue(1)

        # stabilize relative to?
        self.stabilize_to = QComboBox()
        self.stabilize_to.addItem("relative to previous frame")
        self.stabilize_to.addItem("relative to absolute frame")

        # stabilize frame
        self.stabilize_frame = QSpinBox()
        self.stabilize_frame.setMinimum(0)

        # def update_stabilize_frame(i):
        #     tmp = self.stabilize_frame.value()
        #     self.stabilize_frame.setMaximum(
        #         self.viewer.layers[i].data.shape[0] - 1
        #     )
        #     self.stabilize_frame.setValue(tmp)

        # self.input_layer.currentIndexChanged.connect(update_stabilize_frame)

        # increment
        self.upsample_box = QSpinBox()
        self.upsample_box.setMinimum(1)
        self.upsample_box.setMaximum(20)
        self.upsample_box.setValue(1)

        self.channel_select_spin = QSpinBox()
        self.channel_select_spin.setMinimum(0)
        self.channel_select_spin.setMaximum(16)
        self.channel_select_spin.setValue(0)

        # tracking panel
        parameter_panel = QGroupBox("Parameters")
        parameter_layout = QGridLayout()
        i = 1

        parameter_layout.addWidget(QLabel("Channel: "), i, 0)
        parameter_layout.addWidget(self.channel_select_spin, i, 1)
        i += 1

        parameter_layout.addWidget(QLabel("Use ROI: "), i, 0)
        parameter_layout.addWidget(self.roi_checkbox, i, 1)
        i += 1

        parameter_layout.addWidget(QLabel("Roi Z-min"), i, 0)
        parameter_layout.addWidget(self.roi_z_min_spin, i, 1)
        i += 1

        parameter_layout.addWidget(QLabel("Roi Z-max"), i, 0)
        parameter_layout.addWidget(self.roi_z_max_spin, i, 1)
        i += 1

        parameter_layout.addWidget(QLabel("Use 3D: "), i, 0)
        parameter_layout.addWidget(self.use_3d_checkbox, i, 1)
        i += 1

        parameter_layout.addWidget(QLabel("Increment"), i, 0)
        parameter_layout.addWidget(self.increment_box, i, 1)
        i += 1

        parameter_layout.addWidget(QLabel("Align relative to: "), i, 0)
        parameter_layout.addWidget(self.stabilize_to, i, 1)
        i += 1

        parameter_layout.addWidget(QLabel("Fixed frame"), i, 0)
        parameter_layout.addWidget(self.stabilize_frame, i, 1)
        i += 1

        parameter_layout.addWidget(QLabel("Upsample factor"), i, 0)
        parameter_layout.addWidget(self.upsample_box, i, 1)
        i += 1

        parameter_panel.setLayout(parameter_layout)
        self.main_layout.addWidget(parameter_panel)

        # set the layout
        self.main_layout.setAlignment(Qt.AlignTop)

        self.setLayout(self.main_layout)

        def add_output_layer(image, editable=False):
            seg_layer = self.viewer.add_image(image, name="Stabilized")
            seg_layer.editable = editable

        def check_and_run():
            # get image
            image = self.input_image_layers[
                self.input_layer.currentIndex()
            ].data

            # get axis
            dims = ""
            for d, combo in self.axis_combos.items():
                dims += "tcz"[combo.currentIndex()]
            dims += "yx"

            if "t" not in dims:
                raise RuntimeError("Stabilize axis 'Time' not set...")

            if len(set(dims)) != len(dims):
                raise RuntimeError("Dimensions are not unique...")

            # fixed frame
            t0 = self.stabilize_frame.value()

            # use 3d
            use_3d = self.use_3d_checkbox.isChecked()

            # relative or absolute
            stabilize_mode = ["relative", "absolute"][
                self.stabilize_to.currentIndex()
            ]

            # channel
            channel = self.channel_select_spin.value()

            # get increment
            increment = self.increment_box.value()

            # get increment
            upsample_factor = self.upsample_box.value()

            use_roi = self.roi_checkbox.isChecked()

            roi = None
            if use_roi:
                shape_poly = self.viewer.layers["Shapes"].data[0]
                roi = ROIRect.from_shape_poly(
                    shape_poly,
                    dims,
                    z_min=self.roi_z_min_spin.value(),
                    z_max=self.roi_z_max_spin.value(),
                )

            worker = create_worker(
                istabilize,
                image,
                dims,
                use_3d=use_3d,
                roi=roi,
                channel=channel,
                t0=t0,
                increment=increment,
                stabilize_mode=stabilize_mode,
                upsample_factor=upsample_factor,
            )
            worker.returned.connect(add_output_layer)
            worker.start()

        self.run_button.clicked.connect(check_and_run)

    def get_image_dims(self):
        if len(self.viewer.layers) > 0:
            return self.viewer.layers[
                self.input_layer.currentIndex()
            ].data.shape
        else:
            return (0, 0)

    def check_and_get_extra_dims(self):
        *extra_dims, y_dim, x_dim = self.get_image_dims()
        if len(extra_dims) == 0:
            raise RuntimeError("Image is 2D. Nothing to stabilize")
        elif len(extra_dims) > 3:
            raise RuntimeError("Image is 6D. Unable to stabilize")

        return extra_dims
