"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from napari.layers.image.image import Image as IMAGE_LAYER
from napari.layers.shapes.shapes import Shapes as SHAPE_LAYER
from numpy.lib.arraysetops import isin
from qtpy.QtCore import Qt
from qtpy.QtCore import QAbstractTableModel
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    QTableWidgetItem,
    QTableView,
)

from ._core import ISTabilizer, ROIRect
from pandas import DataFrame, read_csv
import numpy as np

if TYPE_CHECKING:
    import napari


class CorrectDriftDock(QWidget):
    ROI_LAYER_NAME = "ROI"

    def _init_input_layer_selection(self):
        ### Input Image layer selection

        self.input_layer = QComboBox()
        self.input_image_layers = []

        def init_input_layer():
            self.input_layer.clear()
            self.input_image_layers = []
            for layer in self.viewer.layers:
                if isinstance(layer, IMAGE_LAYER):
                    self.input_image_layers.append(layer)
                    self.input_layer.addItem(layer.name)

        init_input_layer()

        def on_inserted_layer(event):
            layer = event.value

            if isinstance(layer, IMAGE_LAYER):
                self.input_image_layers.append(layer)
                self.input_layer.addItem(layer.name)

        self.viewer.layers.events.inserted.connect(on_inserted_layer)

        def on_removed_layer(event):
            print("removed", event.value)
            print("current selected", self.get_current_input_layer().name)
            print(type(event.value), type(self.get_current_input_layer().name))

            if event.value == self.get_current_input_layer():
                init_input_layer()

        self.viewer.layers.events.removed.connect(on_removed_layer)

        ### Axes selection

        axis_panel = QWidget()
        axis_layout = QGridLayout()
        axis_panel.setLayout(axis_layout)

        self.axis_combos = {}

        def update_axes_selection(event=None):
            layer = self.get_current_input_layer()

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
                axis_layout.addWidget(QLabel(f"Axis {d} is: "), d + 1, 0)
                axis_layout.addWidget(combo, d + 1, 1)

        self.input_layer.currentIndexChanged.connect(update_axes_selection)

        ### add both to dock
        input_panel = QGroupBox("Input & Axes")
        input_layout = QVBoxLayout()
        input_panel.setLayout(input_layout)

        input_layout.addWidget(self.input_layer)
        input_layout.addWidget(axis_panel)

        self.main_layout.addWidget(input_panel)

    def _init_run_panel(self):
        ### add both to dock
        run_panel = QGroupBox("Correct Drift")
        run_layout = QVBoxLayout()
        run_panel.setLayout(run_layout)

        tmp_panel = QWidget()
        tmp_layout = QHBoxLayout()
        tmp_panel.setLayout(tmp_layout)

        # estimate button
        self.estimate_drift_button = QPushButton("Estimate Drift")
        tmp_layout.addWidget(self.estimate_drift_button)
        self.estimate_drift_button.clicked.connect(self.estimate_drift)

        # tmp_layout.addWidget(QLabel("  or "))

        # load button
        self.load_drift_button = QPushButton("Load Drift")
        tmp_layout.addWidget(self.load_drift_button)
        self.load_drift_button.clicked.connect(self.load_drift)

        run_layout.addWidget(tmp_panel)

        # apply
        self.apply_drift_button = QPushButton("Apply Drift")
        tmp_layout.addWidget(self.apply_drift_button)
        self.apply_drift_button.clicked.connect(self.apply_drift)

        self.main_layout.addWidget(run_panel)

    def get_current_input_layer(self):
        return self.input_image_layers[self.input_layer.currentIndex()]

    def _init_key_and_roi_panel(self):
        # tracking panel
        key_and_roi_panel = QGroupBox("Key frames")
        key_and_roi_layout = QGridLayout()
        key_and_roi_panel.setLayout(key_and_roi_layout)
        i = 1

        # stabilize relative to
        self.estimate_drift_type = QComboBox()
        self.estimate_drift_type.addItem("previous frame")
        self.estimate_drift_type.addItem("absolute frame")

        key_and_roi_layout.addWidget(QLabel("Relative to: "), i, 0)
        key_and_roi_layout.addWidget(self.estimate_drift_type, i, 1)
        i += 1

        # key frame
        self.key_frame = QSpinBox()
        self.key_frame.setMinimum(0)
        self.key_frame.setValue(0)

        key_and_roi_layout.addWidget(QLabel("Key frame"), i, 0)
        key_and_roi_layout.addWidget(self.key_frame, i, 1)
        i += 1

        # key channel
        self.key_channel = QSpinBox()
        self.key_channel.setMinimum(0)
        self.key_channel.setValue(0)

        key_and_roi_layout.addWidget(QLabel("Key channel"), i, 0)
        key_and_roi_layout.addWidget(self.key_channel, i, 1)
        i += 1

        # use roi
        self.roi_checkbox = QCheckBox()
        self.roi_checkbox.setChecked(False)

        def add_shapes_layer(checked):
            if checked:
                for layer in self.viewer.layers:
                    if isinstance(layer, SHAPE_LAYER) and (
                        layer.name == self.ROI_LAYER_NAME
                    ):
                        break
                else:
                    ndim = self.get_current_input_layer().ndim
                    self.viewer.add_shapes(name=self.ROI_LAYER_NAME, ndim=ndim)

        self.roi_checkbox.toggled.connect(add_shapes_layer)

        key_and_roi_layout.addWidget(QLabel("Use ROI: "), i, 0)
        key_and_roi_layout.addWidget(self.roi_checkbox, i, 1)
        i += 1

        # ROI z-min and z=max
        self.roi_z_min = QSpinBox()
        self.roi_z_min.setMinimum(0)
        self.roi_z_min.setValue(0)

        self.roi_z_max = QSpinBox()
        self.roi_z_max.setMinimum(0)
        self.roi_z_max.setValue(0)

        key_and_roi_layout.addWidget(QLabel("Roi Z-min"), i, 0)
        key_and_roi_layout.addWidget(self.roi_z_min, i, 1)
        i += 1

        key_and_roi_layout.addWidget(QLabel("Roi Z-max"), i, 0)
        key_and_roi_layout.addWidget(self.roi_z_max, i, 1)
        i += 1

        self.main_layout.addWidget(key_and_roi_panel)

    def __init__(self, napari_viewer: "napari.Viewer"):
        self.viewer = napari_viewer
        super().__init__()

        self.main_layout = QVBoxLayout()

        self._init_input_layer_selection()

        self._init_run_panel()

        self._init_key_and_roi_panel()

        self._init_other_params()

        self.main_layout.setAlignment(Qt.AlignTop)

        self.setLayout(self.main_layout)

    def _init_other_params(self):
        parameter_panel = QGroupBox("Parameters")
        parameter_layout = QGridLayout()
        parameter_panel.setLayout(parameter_layout)
        i = 1

        # increment
        self.increment_box = QSpinBox()
        self.increment_box.setMinimum(1)
        self.increment_box.setValue(1)

        parameter_layout.addWidget(QLabel("Increment"), i, 0)
        parameter_layout.addWidget(self.increment_box, i, 1)
        i += 1

        # subpixel
        self.upsample_box = QSpinBox()
        self.upsample_box.setMinimum(1)
        self.upsample_box.setMaximum(20)
        self.upsample_box.setValue(1)

        parameter_layout.addWidget(QLabel("Upsample factor"), i, 0)
        parameter_layout.addWidget(self.upsample_box, i, 1)
        i += 1

        # subpixel
        self.extend_output = QCheckBox()
        self.extend_output.setChecked(False)

        parameter_layout.addWidget(QLabel("Extend output"), i, 0)
        parameter_layout.addWidget(self.extend_output, i, 1)
        i += 1

        parameter_panel.setLayout(parameter_layout)
        self.main_layout.addWidget(parameter_panel)

    def _check_input(self):
        layer = self.get_current_input_layer()
        image = layer.data

        # get axis
        dims = ""
        for _, combo in self.axis_combos.items():
            dims += "tcz"[combo.currentIndex()]
        dims += "yx"

        if "t" not in dims:
            raise RuntimeError("Drift correction axis 'Time' not set...")

        if len(set(dims)) != len(dims):
            raise RuntimeError(
                "Dimensions are not unique. Choose each dimensions only once"
            )

        self.ist = ISTabilizer(
            image,
            dims,
        )

        return self.ist

    def estimate_drift(self):
        self._check_input()

        # fixed frame
        key_frame = self.key_frame.value()

        # relative or absolute
        estimate_mode = ["relative", "absolute"][
            self.estimate_drift_type.currentIndex()
        ]

        # channel
        key_channel = self.key_channel.value()

        # get increment
        increment = self.increment_box.value()

        # get increment
        upsample_factor = self.upsample_box.value()

        use_roi = self.roi_checkbox.isChecked()

        roi = None
        if use_roi:
            roi_layer = self.viewer.layers[self.ROI_LAYER_NAME]

            if len(roi_layer.data) == 0:
                raise RuntimeWarning(
                    f"Use ROI checked, but not ROI in '{self.ROI_LAYER_NAME}' shapes layer"
                )

            if len(roi_layer.data) > 1:
                print(
                    "Warning: More than one ROI in shpae layer, using first..."
                )

            shape_poly = roi_layer.data[0]
            roi = ROIRect.from_shape_poly(
                shape_poly,
                self.ist.dims,
                z_min=self.roi_z_min_spin.value(),
                z_max=self.roi_z_max_spin.value(),
            )

        if estimate_mode == "absolute":
            esitmate_offsets_func = self.ist.estimate_shifts_absolute

        else:
            esitmate_offsets_func = self.ist.estimate_shifts_relative

        drift_shifts = esitmate_offsets_func(
            t0=key_frame,
            channel=key_channel,
            increment=increment,
            upsample_factor=upsample_factor,
            roi=roi,
        )

        self.drift_table = TableWidget(self.viewer, self.ist)
        self.drift_table.set_content(drift_shifts)
        # # add widget to napari

        self.viewer.window.add_dock_widget(
            self.drift_table,
            area="right",
            name="Estimated Drift",
            tabify=False,
        )

    def load_drift(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Save as csv...", ".", "*.csv"
        )

        drift_shifts = read_csv(filename, index_col=0)

        self._check_input()

        self.drift_table = TableWidget(self.viewer, self.ist)
        self.drift_table.set_content(drift_shifts.to_numpy())
        # # add widget to napari

        self.viewer.window.add_dock_widget(
            self.drift_table,
            area="right",
            name=f"Drift ({filename})",
            tabify=False,
        )

    def apply_drift(self):
        image_corrected = self.ist.apply_shifts(
            self.drift_table.get_content(),
            extend_output=self.extend_output.isChecked(),
        )

        img_layer = self.viewer.add_image(
            image_corrected,
            name=f"{self.get_current_input_layer().name} (corr)",
        )
        img_layer.editable = False

        # worker = create_worker(
        #     estimate_drift,
        #     ist,
        #     use_3d=use_3d,
        #     roi=roi,
        #     channel=channel,
        #     t0=t0,
        #     increment=increment,
        #     stabilize_mode=stabilize_mode,
        #     upsample_factor=upsample_factor,
        #     progress={"total", image.shape[0]},
        # )

        # worker.returned.connect(add_output_layer)
        # worker.start()

        # self.run_button.clicked.connect(check_and_run)

    # def get_image_dims(self):
    #     if len(self.viewer.layers) > 0:
    #         return self.viewer.layers[
    #             self.input_layer.currentIndex()
    #         ].data.shape
    #     else:
    #         return (0, 0)

    # def check_and_get_extra_dims(self):
    #     *extra_dims, y_dim, x_dim = self.get_image_dims()
    #     if len(extra_dims) == 0:
    #         raise RuntimeError("Image is 2D. Nothing to stabilize")
    #     elif len(extra_dims) > 3:
    #         raise RuntimeError("Image is 6D. Unable to stabilize")

    #     return extra_dims


class TableWidget(QWidget):
    """
    The table widget represents a table inside napari.
    Tables are just views on `properties` of `layers`.
    """

    def __init__(self, ist: ISTabilizer, viewer: "napari.Viewer" = None):
        super().__init__()

        self.ist = ist
        self._napari_viewer = viewer

        self._view = QTableView()

        copy_button = QPushButton("Copy to clipboard")
        copy_button.clicked.connect(self._copy_clicked)

        save_button = QPushButton("Save as csv...")
        save_button.clicked.connect(self._save_clicked)

        self.setWindowTitle("Estimated drift")
        self.setLayout(QGridLayout())
        action_widget = QWidget()
        action_widget.setLayout(QHBoxLayout())
        action_widget.layout().addWidget(copy_button)
        action_widget.layout().addWidget(save_button)
        self.layout().addWidget(action_widget)
        self.layout().addWidget(self._view)
        # action_widget.layout().setSpacing(3)
        # action_widget.layout().setContentsMargins(0, 0, 0, 0)

    def _save_clicked(self, event=None, filename=None):
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save as csv...", ".", "*.csv"
            )
        DataFrame(self.get_content()).to_csv(filename)

    def _copy_clicked(self):
        DataFrame(self.get_content()).to_clipboard()

    def set_content(self, table: np.array):
        self._table = ShiftTableModel(table)
        self._view.setModel(self._table)
        self._view.resizeColumnsToContents()
        self._view.setCornerButtonEnabled(False)

    def get_content(self) -> np.array:
        """
        Returns the current content of the table
        """
        return self._table._data


class ShiftTableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return "ZYX"[section]

        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return str(section)

        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole or role == Qt.EditRole:
                value = self._data[index.row(), index.column()]
                return str(value)

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            try:
                value = float(value)
            except ValueError:
                return False
            self._data[index.row(), index.column()] = value
            return True
        return False

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
