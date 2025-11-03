import sys
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QPushButton, QWidget, QFileDialog, QSlider, QStatusBar,
                             QGroupBox, QLabel, QComboBox, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor, QPalette, QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MRIViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # data and state
        self.data = None
        self.scan_array = None
        self.sitk_image = None
        self.zoom_level = 1.0
        self.current_colormap = 'gray'

        # === Spacing: Default to 1.0 (no stretching) before loading any file ===
        self.spacing_x = 1.0
        self.spacing_y = 1.0
        self.spacing_z = 1.0
        # ======================================================================

        # === MPR States ===
        self.marking_mode = False
        self.curved_path_points = []
        # ==================

        # crosshair positions (indices)
        self.crosshair_x = 0
        self.crosshair_y = 0
        self.crosshair_z = 0

        self.initUI()
        self.apply_light_mode()

    def apply_light_mode(self):
        """Applies a clean light theme matching the medical visualization app."""
        # QT Palette - Clean white background
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(255, 255, 255))
        palette.setColor(QPalette.WindowText, QColor(26, 26, 26))
        palette.setColor(QPalette.Base, QColor(250, 250, 250))
        palette.setColor(QPalette.AlternateBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipBase, QColor(26, 26, 26))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(26, 26, 26))
        palette.setColor(QPalette.Button, QColor(255, 255, 255))
        palette.setColor(QPalette.ButtonText, QColor(26, 26, 26))
        palette.setColor(QPalette.Highlight, QColor(26, 26, 26))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)

        # Matplotlib Style - Light theme
        light_style = {
            "figure.facecolor": "#fafafa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#e8e8e8",
            "text.color": "#1a1a1a",
            "xtick.color": "#1a1a1a",
            "ytick.color": "#1a1a1a",
            "axes.labelcolor": "#1a1a1a",
            "axes.titlecolor": "#1a1a1a",
            "savefig.facecolor": "#fafafa"
        }
        plt.rcParams.update(light_style)

        # Apply to existing figures
        self.axial_fig.patch.set_facecolor(light_style["figure.facecolor"])
        self.coronal_fig.patch.set_facecolor(light_style["figure.facecolor"])
        self.sagittal_fig.patch.set_facecolor(light_style["figure.facecolor"])
        self.curved_fig.patch.set_facecolor(light_style["figure.facecolor"])

        self.crosshair_color = '#1a1a1a'
        self.update_all_slices()

    def initUI(self):
        self.setWindowTitle("MPR Viewer - Simplified")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("background-color: #ffffff;")

        self.main_layout = QHBoxLayout()
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # -------- Left control panel --------
        self.control_panel = QFrame()
        self.control_panel.setFixedWidth(280)
        self.control_panel.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-right: 1px solid #e8e8e8;
            }
        """)

        self.control_layout = QVBoxLayout()
        self.control_layout.setSpacing(20)
        self.control_layout.setContentsMargins(25, 25, 25, 25)
        self.control_panel.setLayout(self.control_layout)

        # Title
        title_label = QLabel("MPR Viewer")
        title_label.setFont(QFont("SF Pro Display", 18, QFont.Medium))
        title_label.setStyleSheet("color: #1a1a1a;")
        self.control_layout.addWidget(title_label)

        # Load single file button only
        self.load_file_button = self.create_minimal_button('Load NIfTI File')
        self.load_file_button.clicked.connect(self.load_single_file)
        self.control_layout.addWidget(self.load_file_button)

        # Playback
        self.play_pause_button = self.create_minimal_button("Play (Axial Slices)")
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.control_layout.addWidget(self.play_pause_button)

        # Colormap selection
        colormap_group = QGroupBox("Colormap")
        colormap_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #1a1a1a;
                border: 1px solid #e8e8e8;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        colormap_layout = QVBoxLayout()
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet'])
        self.colormap_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #e8e8e8;
                border-radius: 4px;
                padding: 5px;
                background-color: #fafafa;
                color: #1a1a1a;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #1a1a1a;
                margin-right: 5px;
            }
        """)
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        colormap_layout.addWidget(self.colormap_combo)
        colormap_group.setLayout(colormap_layout)
        self.control_layout.addWidget(colormap_group)

        # Reset view
        self.reset_button = self.create_minimal_button("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        self.control_layout.addWidget(self.reset_button)

        # === CURVED MPR CONTROLS ===
        mpr_group = QGroupBox("Curved MPR Tools")
        mpr_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #1a1a1a;
                border: 1px solid #e8e8e8;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        mpr_layout = QVBoxLayout()

        self.mark_path_button = self.create_minimal_button("1. Mark Curved Path")
        self.mark_path_button.setCheckable(True)
        self.mark_path_button.clicked.connect(self.toggle_path_marking)
        mpr_layout.addWidget(self.mark_path_button)

        self.show_curved_planar_button = self.create_minimal_button("2. Show Curved Planar")
        self.show_curved_planar_button.clicked.connect(self.show_curved_planar_view)
        mpr_layout.addWidget(self.show_curved_planar_button)

        self.show_panoramic_button = self.create_minimal_button("3. Show Panoramic MPR")
        self.show_panoramic_button.clicked.connect(self.show_panoramic_view)
        mpr_layout.addWidget(self.show_panoramic_button)

        mpr_group.setLayout(mpr_layout)
        self.control_layout.addWidget(mpr_group)
        # ===============================

        self.control_layout.addStretch()

        # Status bar
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("color: #666666; font-size: 11px;")
        self.status_bar.setWordWrap(True)
        self.control_layout.addWidget(self.status_bar)

        self.main_layout.addWidget(self.control_panel)

        # -------- Right panel (viewports) --------
        self.viewport_panel = QWidget()
        self.viewport_panel.setStyleSheet("background-color: #fafafa;")
        self.viewport_layout = QVBoxLayout()
        self.viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.viewport_layout.setSpacing(0)
        self.viewport_panel.setLayout(self.viewport_layout)

        # Matplotlib figures and canvases
        self.axial_fig, self.axial_ax = plt.subplots()
        self.coronal_fig, self.coronal_ax = plt.subplots()
        self.sagittal_fig, self.sagittal_ax = plt.subplots()
        self.curved_fig, self.curved_ax = plt.subplots()

        self.curved_canvas = FigureCanvas(self.curved_fig)
        self.curved_slider = QSlider(Qt.Horizontal)

        # The slider is visible by default
        self.curved_slider.show()

        self.axial_canvas = FigureCanvas(self.axial_fig)
        self.coronal_canvas = FigureCanvas(self.coronal_fig)
        self.sagittal_canvas = FigureCanvas(self.sagittal_fig)

        # connect events
        self.axial_canvas.mpl_connect('scroll_event', lambda event: self.wheel_control(event, 0))
        self.coronal_canvas.mpl_connect('scroll_event', lambda event: self.wheel_control(event, 1))
        self.sagittal_canvas.mpl_connect('scroll_event', lambda event: self.wheel_control(event, 2))

        self.axial_canvas.mpl_connect('motion_notify_event', self.update_crosshairs)
        self.coronal_canvas.mpl_connect('motion_notify_event', self.update_crosshairs)
        self.sagittal_canvas.mpl_connect('motion_notify_event', self.update_crosshairs)
        self.axial_canvas.mpl_connect('button_press_event', self.update_crosshairs_on_click)
        self.coronal_canvas.mpl_connect('button_press_event', self.update_crosshairs_on_click)
        self.sagittal_canvas.mpl_connect('button_press_event', self.update_crosshairs_on_click)

        # crosshair lines (initialized)
        self.crosshair_color = '#1a1a1a'
        self.axial_vline = self.axial_ax.axvline(0, color=self.crosshair_color, linestyle='--', linewidth=0.8)
        self.axial_hline = self.axial_ax.axhline(0, color=self.crosshair_color, linestyle='--', linewidth=0.8)
        self.coronal_vline = self.coronal_ax.axvline(0, color=self.crosshair_color, linestyle='--', linewidth=0.8)
        self.coronal_hline = self.coronal_ax.axhline(0, color=self.crosshair_color, linestyle='--', linewidth=0.8)
        self.sagittal_vline = self.sagittal_ax.axvline(0, color=self.crosshair_color, linestyle='--', linewidth=0.8)
        self.sagittal_hline = self.sagittal_ax.axhline(0, color=self.crosshair_color, linestyle='--', linewidth=0.8)

        # sliders for slices
        self.axial_slider = QSlider(Qt.Horizontal)
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider = QSlider(Qt.Horizontal)

        # Style sliders
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #e8e8e8;
                height: 4px;
                background: #f5f5f5;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #1a1a1a;
                border: none;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #333333;
            }
        """
        self.axial_slider.setStyleSheet(slider_style)
        self.coronal_slider.setStyleSheet(slider_style)
        self.sagittal_slider.setStyleSheet(slider_style)
        self.curved_slider.setStyleSheet(slider_style)

        self.axial_slider.valueChanged.connect(self.update_axial_slice)
        self.coronal_slider.valueChanged.connect(self.update_coronal_slice)
        self.sagittal_slider.valueChanged.connect(self.update_sagittal_slice)

        # Grid layout for viewports
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)

        self.axial_group = self.create_viewport_group("Axial View (X-Y)", self.axial_canvas, self.axial_slider)
        self.coronal_group = self.create_viewport_group("Coronal View (X-Z)", self.coronal_canvas, self.coronal_slider)
        self.sagittal_group = self.create_viewport_group("Sagittal View (Y-Z)", self.sagittal_canvas,
                                                         self.sagittal_slider)

        # Curved MPR Group (initialized but hidden)
        self.curved_group = self.create_viewport_group("Curved MPR View", self.curved_canvas, self.curved_slider)
        self.curved_group.hide()

        self.grid_layout.addWidget(self.axial_group, 0, 0)
        self.grid_layout.addWidget(self.sagittal_group, 0, 1)
        self.grid_layout.addWidget(self.coronal_group, 1, 0)
        self.grid_layout.addWidget(self.curved_group, 1, 1)

        self.viewport_layout.addLayout(self.grid_layout)
        self.main_layout.addWidget(self.viewport_panel)

        # playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_slices)
        self.is_playing = False

        self.setLayout(self.main_layout)
        self.setFocusPolicy(Qt.StrongFocus)

    def create_minimal_button(self, text):
        """Create minimal button matching the medical visualization app style"""
        btn = QPushButton(text)
        btn.setFixedHeight(40)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-family: 'SF Pro Display';
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #000000;
            }
            QPushButton:checked {
                background-color: #0066cc;
            }
        """)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def create_viewport_group(self, title, canvas, slider):
        group = QFrame()
        group.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e8e8e8;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("SF Pro Display", 12, QFont.Medium))
        title_label.setStyleSheet("color: #1a1a1a; border: none;")
        layout.addWidget(title_label)

        layout.addWidget(canvas)

        slider_container = QWidget()
        slider_container.setStyleSheet("background: transparent; border: none;")
        slider_layout = QVBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addWidget(slider)
        slider_container.setLayout(slider_layout)

        slider_container.setMaximumHeight(30)

        layout.addWidget(slider_container)
        group.setLayout(layout)
        return group

    # --------- Loading ---------

    def load_single_file(self):
        """Loads a single NIfTI (.nii, .nii.gz) file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "",
                                                   "NIfTI Files (*.nii *.nii.gz);;All files (*)")
        if not file_path:
            return

        try:
            self.status_bar.setText(f"⏳ Loading: {os.path.basename(file_path)}...")
            QApplication.processEvents()

            sitk_image = sitk.ReadImage(file_path)
            file_source = os.path.basename(file_path)

            self.process_loaded_image(sitk_image, file_source)

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file: {e}")
            self.status_bar.setText(f"❌ Failed to load file")

    def process_loaded_image(self, sitk_image, file_source):
        """Common function to set up the GUI after any image is loaded."""

        # Check if loaded image is 3D
        if sitk_image.GetDimension() < 3:
            QMessageBox.warning(self, "2D Image Loaded",
                                "You loaded a single 2D slice. \n"
                                "Coronal, Sagittal views will not be meaningful."
                                )
            if sitk_image.GetDimension() == 2:
                sitk_image = sitk.JoinSeries([sitk_image])

        self.sitk_image = sitk_image

        # Load Spacing and Array
        self.scan_array = sitk.GetArrayFromImage(self.sitk_image)

        self.spacing_x, self.spacing_y, self.spacing_z = self.sitk_image.GetSpacing()

        self.status_bar.setText(
            f"✓ Loaded: {file_source}\nShape: {self.scan_array.shape}\nSpacing: ({self.spacing_x:.2f}, {self.spacing_y:.2f}, {self.spacing_z:.2f})"
        )

        # Configure sliders
        self.axial_slider.setMaximum(self.scan_array.shape[0] - 1)
        self.coronal_slider.setMaximum(self.scan_array.shape[1] - 1)
        self.sagittal_slider.setMaximum(self.scan_array.shape[2] - 1)
        self.curved_slider.setRange(0, self.scan_array.shape[0] - 1)

        self.crosshair_x = self.scan_array.shape[2] // 2
        self.crosshair_y = self.scan_array.shape[1] // 2
        self.crosshair_z = self.scan_array.shape[0] // 2

        self.axial_slider.setValue(self.crosshair_z)
        self.coronal_slider.setValue(self.crosshair_y)
        self.sagittal_slider.setValue(self.crosshair_x)

        # Reset Curved MPR path
        self.curved_path_points = []
        self.mark_path_button.setChecked(False)
        self.mark_path_button.setText("1. Mark Curved Path")
        self.curved_group.hide()

        self.reset_view()
        self.update_all_slices()

    # ---------- Crosshairs & clicking ----------

    def toggle_path_marking(self, checked):
        """Toggle the mode for marking points on the axial view."""
        self.marking_mode = checked
        if checked:
            self.curved_path_points = []
            self.mark_path_button.setText("Marking (Click on Axial)")
            self.status_bar.setText("Click on the Axial view to define the curved path.")
        else:
            self.mark_path_button.setText(f"1. Mark Path ({len(self.curved_path_points)} pts)")
            self.status_bar.setText(f"Path has {len(self.curved_path_points)} points.")

        self.update_axial_slice(self.crosshair_z)

    def update_crosshairs(self, event):
        """Updates crosshair lines and status bar on mouse motion/hover."""
        if event.inaxes is None or self.scan_array is None or self.scan_array.size == 0:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        x_int, y_int = int(round(x)), int(round(y))
        z_idx = 0
        val = 0.0

        try:
            if event.inaxes == self.axial_ax:
                z_idx = self.crosshair_z
                if 0 <= z_idx < self.scan_array.shape[0] and 0 <= y_int < self.scan_array.shape[1] and 0 <= x_int < \
                        self.scan_array.shape[2]:
                    val = self.scan_array[z_idx, y_int, x_int]
                    self.axial_vline.set_xdata(x_int)
                    self.axial_hline.set_ydata(y_int)
                    self.axial_canvas.draw_idle()

            elif event.inaxes == self.coronal_ax:
                z_idx = self.scan_array.shape[0] - 1 - y_int
                y_idx = self.crosshair_y
                if 0 <= z_idx < self.scan_array.shape[0] and 0 <= y_idx < self.scan_array.shape[1] and 0 <= x_int < \
                        self.scan_array.shape[2]:
                    val = self.scan_array[z_idx, y_idx, x_int]
                    self.coronal_vline.set_xdata(x_int)
                    self.coronal_hline.set_ydata(y_int)
                    self.coronal_canvas.draw_idle()

            elif event.inaxes == self.sagittal_ax:
                z_idx = self.scan_array.shape[0] - 1 - y_int
                y_idx = x_int
                x_idx = self.crosshair_x
                if 0 <= z_idx < self.scan_array.shape[0] and 0 <= y_idx < self.scan_array.shape[1] and 0 <= x_idx < \
                        self.scan_array.shape[2]:
                    val = self.scan_array[z_idx, y_idx, x_idx]
                    self.sagittal_vline.set_xdata(x_int)
                    self.sagittal_hline.set_ydata(y_int)
                    self.sagittal_canvas.draw_idle()

        except IndexError:
            pass

    def update_crosshairs_on_click(self, event):
        """Update crosshairs when user clicks inside a view OR collects a point."""
        if event.inaxes is None or self.scan_array is None:
            return

        # Curved MPR point collection
        if self.marking_mode and event.inaxes == self.axial_ax and event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.curved_path_points.append((x, y))
            self.status_bar.setText(f"Added point ({x}, {y}). Total: {len(self.curved_path_points)}")
            self.mark_path_button.setText(f"Marking ({len(self.curved_path_points)} pts)")
            self.update_axial_slice(self.crosshair_z)
            return

        try:
            xlim, ylim = event.inaxes.get_xlim(), event.inaxes.get_ylim()
        except Exception:
            xlim, ylim = None, None

        if event.inaxes == self.axial_ax and event.xdata is not None and event.ydata is not None:
            self.crosshair_x = int(event.xdata)
            self.crosshair_y = int(event.ydata)
            self.axial_slider.setValue(self.crosshair_z)
            self.sagittal_slider.setValue(self.crosshair_x)
            self.coronal_slider.setValue(self.crosshair_y)

        elif event.inaxes == self.coronal_ax and event.xdata is not None and event.ydata is not None:
            self.crosshair_x = int(event.xdata)
            self.crosshair_z = int(self.scan_array.shape[0] - 1 - event.ydata)
            self.sagittal_slider.setValue(self.crosshair_x)
            self.axial_slider.setValue(self.crosshair_z)

        elif event.inaxes == self.sagittal_ax and event.xdata is not None and event.ydata is not None:
            self.crosshair_y = int(event.xdata)
            self.crosshair_z = int(self.scan_array.shape[0] - 1 - event.ydata)
            self.coronal_slider.setValue(self.crosshair_y)
            self.axial_slider.setValue(self.crosshair_z)

        self.update_all_slices()

        if xlim is not None and ylim is not None:
            event.inaxes.set_xlim(xlim)
            event.inaxes.set_ylim(ylim)
            event.canvas.draw_idle()

    # ---------- Display update functions ----------

    def update_axial_slice(self, value):
        self.crosshair_z = int(value)
        if self.scan_array is not None:
            self.show_axial_slice(self.scan_array, self.crosshair_z)

            if self.curved_group.isVisible() and self.curved_slider.isVisible():
                self.perform_curved_planar_mpr(value)

    def update_coronal_slice(self, value):
        self.crosshair_y = int(value)
        if self.scan_array is not None:
            self.show_coronal_slice(self.scan_array, self.crosshair_y)

    def update_sagittal_slice(self, value):
        self.crosshair_x = int(value)
        if self.scan_array is not None:
            self.show_sagittal_slice(self.scan_array, self.crosshair_x)

    def update_all_slices(self):
        if self.scan_array is not None:
            self.update_axial_slice(self.crosshair_z)
            self.update_coronal_slice(self.crosshair_y)
            self.update_sagittal_slice(self.crosshair_x)

    def show_axial_slice(self, scan, slice_index):
        if scan is None or scan.ndim < 3 or slice_index >= scan.shape[0]:
            return
        self.axial_ax.clear()
        slice_data = scan[int(slice_index), :, :]
        aspect_ratio = self.spacing_y / self.spacing_x
        self.display_slice(self.axial_ax, slice_data, "Axial View (X-Y)", aspect_ratio)

        # Draw Path Points
        if self.curved_path_points:
            x_coords, y_coords = zip(*self.curved_path_points)
            self.axial_ax.plot(x_coords, y_coords, 'o-', color='#0066cc', markersize=6, linewidth=2, alpha=0.8)

        self.axial_vline = self.axial_ax.axvline(self.crosshair_x, color=self.crosshair_color, linestyle='--',
                                                 linewidth=0.8)
        self.axial_hline = self.axial_ax.axhline(self.crosshair_y, color=self.crosshair_color, linestyle='--',
                                                 linewidth=0.8)
        self.axial_ax.plot(self.crosshair_x, self.crosshair_y, 'o', color=self.crosshair_color, markersize=4)
        self.axial_canvas.draw_idle()

    def show_coronal_slice(self, scan, slice_index):
        if scan is None or scan.ndim < 3 or slice_index >= scan.shape[1]:
            return
        self.coronal_ax.clear()
        slice_data = scan[:, int(slice_index), :]
        slice_data_flipped = np.flipud(slice_data)

        aspect_ratio = self.spacing_z / self.spacing_x
        self.display_slice(self.coronal_ax, slice_data_flipped, "Coronal View (X-Z)", aspect_ratio)

        z_line_y_coord = self.scan_array.shape[0] - 1 - self.crosshair_z
        self.coronal_vline = self.coronal_ax.axvline(self.crosshair_x, color=self.crosshair_color, linestyle='--',
                                                     linewidth=0.8)
        self.coronal_hline = self.coronal_ax.axhline(z_line_y_coord, color=self.crosshair_color, linestyle='--',
                                                     linewidth=0.8)
        self.coronal_ax.plot(self.crosshair_x, z_line_y_coord, 'o', color=self.crosshair_color, markersize=4)
        self.coronal_canvas.draw_idle()

    def show_sagittal_slice(self, scan, slice_index):
        if scan is None or scan.ndim < 3 or slice_index >= scan.shape[2]:
            return
        self.sagittal_ax.clear()
        slice_data = scan[:, :, int(slice_index)]
        slice_data_flipped = np.flipud(slice_data)

        aspect_ratio = self.spacing_z / self.spacing_y
        self.display_slice(self.sagittal_ax, slice_data_flipped, "Sagittal View (Y-Z)", aspect_ratio)

        z_line_y_coord = self.scan_array.shape[0] - 1 - self.crosshair_z
        self.sagittal_vline = self.sagittal_ax.axvline(self.crosshair_y, color=self.crosshair_color, linestyle='--',
                                                       linewidth=0.8)
        self.sagittal_hline = self.sagittal_ax.axhline(z_line_y_coord, color=self.crosshair_color, linestyle='--',
                                                       linewidth=0.8)
        self.sagittal_ax.plot(self.crosshair_y, z_line_y_coord, 'o', color=self.crosshair_color, markersize=4)
        self.sagittal_canvas.draw_idle()

    def display_slice(self, ax, slice_data, title, aspect_ratio=None):
        """Display slice with clean styling."""
        if slice_data is None:
            return

        minv = np.min(slice_data)
        maxv = np.max(slice_data)

        if maxv == minv:
            normalized_data = np.zeros_like(slice_data, dtype=np.float32)
        else:
            normalized_data = (slice_data - minv).astype(np.float32) / (maxv - minv)

        display_data = (normalized_data * 255).astype(np.uint8)

        # Clean matplotlib styling
        ax.tick_params(colors='#1a1a1a', labelsize=8)
        ax.title.set_color('#1a1a1a')
        for spine in ax.spines.values():
            spine.set_color('#e8e8e8')
            spine.set_linewidth(0.5)

        ax.imshow(display_data, cmap=self.current_colormap, origin='upper',
                  aspect=aspect_ratio if aspect_ratio is not None else 'auto')

        ax.set_title(title, fontsize=10, pad=8)
        ax.axis('off')

    # ---------- Curved MPR Logic ----------

    def update_curved_planar_slice(self, value):
        """Callback for the Curved Planar slider to change the Z-plane."""
        self.perform_curved_planar_mpr(value)

    def show_curved_planar_view(self):
        """Activates the Z-Slicing 'Curved Planar' view."""
        if self.scan_array is None or len(self.curved_path_points) < 2:
            self.status_bar.setText("Load an image and mark at least 2 points first.")
            return

        self.curved_group.show()
        self.curved_slider.show()

        self.curved_slider.setValue(self.crosshair_z)
        try:
            self.curved_slider.valueChanged.disconnect()
        except TypeError:
            pass
        self.curved_slider.valueChanged.connect(self.update_curved_planar_slice)

        self.perform_curved_planar_mpr(self.crosshair_z)

    def show_panoramic_view(self):
        """Activates the full-height 'Panoramic' view."""
        if self.scan_array is None or len(self.curved_path_points) < 2:
            self.status_bar.setText("Load an image and mark at least 2 points first.")
            return

        self.curved_group.show()
        self.curved_slider.hide()

        self.perform_panoramic_mpr()

    def perform_curved_planar_mpr(self, z_slice_index):
        """Calculates and displays the Curved MPR slice AT A SPECIFIC Z-INDEX."""
        if self.scan_array is None or len(self.curved_path_points) < 2:
            return

        # Path Interpolation
        points = np.array(self.curved_path_points)
        x_pts, y_pts = points[:, 0], points[:, 1]

        num_path_points = 800

        t = np.linspace(0, 1, len(x_pts))
        t_fine = np.linspace(0, 1, num_path_points)

        try:
            interp_x = interp1d(t, x_pts, kind='cubic')
            interp_y = interp1d(t, y_pts, kind='cubic')
            path_x = interp_x(t_fine)
            path_y = interp_y(t_fine)
        except ValueError:
            path_x = np.interp(t_fine, t, x_pts)
            path_y = np.interp(t_fine, t, y_pts)

        # Calculate normals
        dx = np.gradient(path_x)
        dy = np.gradient(path_y)
        normal_x = -dy
        normal_y = dx

        magnitude = np.sqrt(normal_x ** 2 + normal_y ** 2)
        magnitude[magnitude == 0] = 1
        normal_x /= magnitude
        normal_y /= magnitude

        # Create grid
        slice_width_voxels = 330
        slice_range = np.linspace(-slice_width_voxels // 2, slice_width_voxels // 2, slice_width_voxels)

        Z_coords = np.ones((slice_width_voxels, num_path_points)) * float(z_slice_index)
        Y_coords = path_y[np.newaxis, :] + slice_range[:, np.newaxis] * normal_y[np.newaxis, :]
        X_coords = path_x[np.newaxis, :] + slice_range[:, np.newaxis] * normal_x[np.newaxis, :]

        coords = np.array([Z_coords, Y_coords, X_coords])

        # Resample
        curved_mpr_slice = map_coordinates(self.scan_array, coords, order=1, cval=np.min(self.scan_array))

        # Display
        self.curved_ax.clear()
        self.display_slice(self.curved_ax, curved_mpr_slice, f"Curved Planar (Z={int(z_slice_index)})", None)
        self.curved_ax.set_xlabel('Length along Curve', fontsize=9)
        self.curved_ax.set_ylabel('Perpendicular Axis', fontsize=9)
        self.curved_canvas.draw_idle()
        self.status_bar.setText(f"Curved Planar at Z={int(z_slice_index)}")

    def perform_panoramic_mpr(self):
        """Calculates and displays a panoramic curved MPR."""
        if self.scan_array is None or len(self.curved_path_points) < 2:
            return

        # Path Interpolation
        points = np.array(self.curved_path_points)
        x_pts, y_pts = points[:, 0], points[:, 1]

        num_path_points = 800

        t = np.linspace(0, 1, len(x_pts))
        t_fine = np.linspace(0, 1, num_path_points)

        try:
            interp_x = interp1d(t, x_pts, kind='cubic')
            interp_y = interp1d(t, y_pts, kind='cubic')
            path_x = interp_x(t_fine)
            path_y = interp_y(t_fine)
        except ValueError:
            path_x = np.interp(t_fine, t, x_pts)
            path_y = np.interp(t_fine, t, y_pts)

        # Create grid
        num_z_points = self.scan_array.shape[0]
        z_indices = np.arange(num_z_points)

        Z_coords = np.tile(z_indices[:, np.newaxis], (1, num_path_points))
        Y_coords = np.tile(path_y[np.newaxis, :], (num_z_points, 1))
        X_coords = np.tile(path_x[np.newaxis, :], (num_z_points, 1))

        coords = np.array([Z_coords, Y_coords, X_coords])

        # Resample
        curved_mpr_slice = map_coordinates(self.scan_array, coords, order=1, cval=np.min(self.scan_array))

        # Display
        self.curved_ax.clear()
        aspect_ratio = self.spacing_z / self.spacing_x
        self.display_slice(self.curved_ax, np.flipud(curved_mpr_slice), "Panoramic MPR View", aspect_ratio)
        self.curved_ax.set_xlabel('Length along Curve', fontsize=9)
        self.curved_ax.set_ylabel('Z-Axis (Height)', fontsize=9)
        self.curved_canvas.draw_idle()
        self.status_bar.setText(f"Panoramic MPR generated ({len(self.curved_path_points)} points)")

    # ---------- Slicing / Zooming / Panning Controls ----------

    def wheel_control(self, event, view_index):
        """Controls slicing (default) or zooming (Ctrl + wheel)."""
        if event.inaxes is None or self.scan_array is None:
            return

        is_zoom = event.key in ('control', 'ctrl')
        delta = 1 if event.button == 'up' else -1

        if is_zoom:
            # Zoom Logic
            ax = event.inaxes
            base_scale = 1.1
            scale_factor = base_scale if event.button == 'up' else 1.0 / base_scale

            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_data, y_data = event.xdata, event.ydata
            if x_data is None or y_data is None:
                return
            x_range, y_range = x_max - x_min, y_max - y_min
            rel_x, rel_y = (x_data - x_min) / x_range, (y_data - y_min) / y_range

            new_x_range, new_y_range = x_range / scale_factor, y_range / scale_factor
            new_x_min = x_data - rel_x * new_x_range
            new_x_max = new_x_min + new_x_range
            new_y_min = y_data - rel_y * new_y_range
            new_y_max = new_y_min + new_y_range

            ax.set_xlim(new_x_min, new_x_max)
            ax.set_ylim(new_y_min, new_y_max)
            ax.figure.canvas.draw_idle()

        else:
            # Slicing Logic
            if view_index == 0:
                new_z = np.clip(self.crosshair_z + delta, 0, self.scan_array.shape[0] - 1)
                self.axial_slider.setValue(new_z)

            elif view_index == 1:
                new_y = np.clip(self.crosshair_y + delta, 0, self.scan_array.shape[1] - 1)
                self.coronal_slider.setValue(new_y)

            elif view_index == 2:
                new_x = np.clip(self.crosshair_x + delta, 0, self.scan_array.shape[2] - 1)
                self.sagittal_slider.setValue(new_x)

    def toggle_playback(self):
        if self.is_playing:
            self.playback_timer.stop()
            self.play_pause_button.setText("Play (Axial Slices)")
        else:
            self.playback_timer.start(30)
            self.play_pause_button.setText("Pause")
        self.is_playing = not self.is_playing

    def update_slices(self):
        if not self.is_playing or self.scan_array is None:
            return
        new_val = self.axial_slider.value() + 1
        if new_val > self.axial_slider.maximum():
            new_val = 0
        self.axial_slider.setValue(new_val)

    def reset_view(self):
        if self.scan_array is None:
            self.axial_ax.clear()
            self.coronal_ax.clear()
            self.sagittal_ax.clear()
            self.curved_ax.clear()
            self.axial_ax.set_title("Axial View (X-Y)")
            self.coronal_ax.set_title("Coronal View (X-Z)")
            self.sagittal_ax.set_title("Sagittal View (Y-Z)")
            self.curved_ax.set_title("Curved MPR View")
            self.axial_canvas.draw_idle()
            self.coronal_canvas.draw_idle()
            self.sagittal_canvas.draw_idle()
            self.curved_canvas.draw_idle()
            return

        self.crosshair_x = self.scan_array.shape[2] // 2
        self.crosshair_y = self.scan_array.shape[1] // 2
        self.crosshair_z = self.scan_array.shape[0] // 2

        self.axial_slider.setValue(self.crosshair_z)
        self.coronal_slider.setValue(self.crosshair_y)
        self.sagittal_slider.setValue(self.crosshair_x)

        self.axial_ax.autoscale(enable=True, axis='both', tight=True)
        self.coronal_ax.autoscale(enable=True, axis='both', tight=True)
        self.sagittal_ax.autoscale(enable=True, axis='both', tight=True)
        self.curved_ax.autoscale(enable=True, axis='both', tight=True)

        self.axial_ax.set_aspect(self.spacing_y / self.spacing_x)
        self.coronal_ax.set_aspect(self.spacing_z / self.spacing_x)
        self.sagittal_ax.set_aspect(self.spacing_z / self.spacing_y)

        self.current_colormap = 'gray'
        self.colormap_combo.setCurrentText('gray')
        self.update_all_slices()
        self.status_bar.setText("✓ View reset to default")

    def keyPressEvent(self, event):
        step_size = 10
        if event.key() == Qt.Key_Left:
            self.pan_view(-step_size, 0)
        elif event.key() == Qt.Key_Right:
            self.pan_view(step_size, 0)
        elif event.key() == Qt.Key_Up:
            self.pan_view(0, -step_size)
        elif event.key() == Qt.Key_Down:
            self.pan_view(0, step_size)

    def pan_view(self, dx, dy):
        mouse_pos = QApplication.instance().widgetAt(QCursor.pos())
        if mouse_pos == self.axial_canvas:
            self.pan_specific_view(self.axial_ax, dx, dy)
        elif mouse_pos == self.coronal_canvas:
            self.pan_specific_view(self.coronal_ax, dx, dy)
        elif mouse_pos == self.sagittal_canvas:
            self.pan_specific_view(self.sagittal_ax, dx, dy)
        elif mouse_pos == self.curved_canvas:
            self.pan_specific_view(self.curved_ax, dx, dy)

    def pan_specific_view(self, ax, dx, dy):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        ax.figure.canvas.draw_idle()

    def update_colormap(self, colormap_name):
        self.current_colormap = colormap_name
        self.update_all_slices()