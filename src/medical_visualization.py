import re
import vtk
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                             QFileDialog, QMessageBox, QStackedWidget, QGridLayout,
                             QGraphicsDropShadowEffect, QProgressBar, QSpinBox,
                             QColorDialog, QScrollArea, QSlider,QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPixmap, QPainter
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import sys
import os
import glob
from pathlib import Path
import pickle
import hashlib
import traceback

# Import the MRI Viewer and Camera Fly-Through
from mri_viewer import MRIViewer
from camera_flythrough import CameraAnimator, CustomInteractorStyle, PathVisualizer, ClippingPlaneManager

try:
    import nibabel as nib
    from skimage import measure
    import numpy as np

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("ERROR: Required libraries not installed. Run:")
    print("pip install nibabel scikit-image numpy scipy")


class MedicalVisualization(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_actors = []
        self.current_system = None

        # MRI Viewer window reference
        self.mri_viewer_window = None

        # Animation variables
        self.animation_timer = None
        self.current_frame = 0
        self.preloaded_frames = []
        self.total_frames = 0
        self.is_animating = False

        # Clipping planes state
        self.clip_planes_enabled = False
        self.clip_plane_sagittal = vtk.vtkPlane()
        self.clip_plane_coronal = vtk.vtkPlane()
        self.clip_plane_axial = vtk.vtkPlane()
        self.clip_plane_collection = vtk.vtkPlaneCollection()
        self.bounds = None

        # --- Visual clipping planes ---
        self.plane_source_sagittal = None
        self.plane_actor_sagittal = None
        self.plane_source_coronal = None
        self.plane_actor_coronal = None
        self.plane_source_axial = None
        self.plane_actor_axial = None

        # Fly-through variables
        self.path_definition_mode = False
        self.fly_path_points = []
        self.path_actor = None
        self.camera_animator = None
        self.flythrough_timer = None
        self.is_flying = False
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.clipping_manager = None

        # UPDATE THESE PATHS to your NIfTI folders
        self.organ_folders = {
            "Nervous": "/Users/rahma/Desktop/drivefolders/brain",
            "Cardiovascular": "/Users/rahma/Desktop/drivefolders/heart",
            "Musculoskeletal": "/Users/rahma/Desktop/drivefolders/bones",
            "Dental": "C:/Users/Youssef/Desktop/Mpr visualization/Dataset/Teeth"
        }

        # Path to heart animation STL frames
        self.heart_animation_folder = "Dataset/heartmv"
        # Updated color scheme (replace the existing one in __init__)
        self.color_schemes = {
            "Nervous": {
                "frontal": (0.2, 0.5, 0.9),
                "parietal": (0.3, 0.8, 0.4),
                "temporal": (0.95, 0.6, 0.2),
                "occipital": (0.8, 0.3, 0.8),
                "limbic": (0.9, 0.3, 0.3),
                "cerebellum": (0.3, 0.9, 0.9),
                "insular": (0.9, 0.9, 0.3),
                "brainstem": (0.7, 0.5, 0.3),
                "vermis": (0.5, 0.8, 0.6),
                "default": (0.7, 0.7, 0.7)
            },
            "Cardiovascular": {
                "atrium": (0.95, 0.6, 0.4),
                "ventricle": (0.8, 0.1, 0.1),
                "myocardium": (0.7, 0.15, 0.15),
                "aorta": (0.9, 0.6, 0.2),
                "pulmonary": (0.3, 0.6, 1.0),
                "vein": (0.3, 0.4, 0.8),
                "artery": (0.9, 0.3, 0.3),
                "coronary": (1.0, 0.2, 0.2),
                "vena_cava": (0.3, 0.4, 0.8),
                "iliac": (0.7, 0.5, 0.3),
                "subclavian": (0.9, 0.3, 0.3),
                "superior_vena": (0.3, 0.4, 0.8),
                "default": (0.8, 0.3, 0.3)
            },
            "Musculoskeletal": {
                "skull": (0.9, 0.9, 0.85),
                "spine": (0.85, 0.85, 0.8),
                "rib": (0.88, 0.88, 0.83),
                "femur": (0.92, 0.92, 0.87),
                "pelvis": (0.87, 0.87, 0.82),
                "default": (0.9, 0.9, 0.85)
            },
            "Dental": {
                "pulp": (0.86, 0.08, 0.24),  # Pink/Red - MUST BE FIRST
                "jawbone": (0.96, 0.96, 0.86),  # Bone white/ivory
                "tooth": (1.0, 1.0, 0.94),  # Enamel white
                "molar": (1.0, 1.0, 0.94),  # Enamel white
                "incisor": (1.0, 1.0, 0.94),  # Enamel white
                "canine": (1.0, 1.0, 0.94),  # Enamel white
                "premolar": (1.0, 1.0, 0.94),  # Enamel white
                "pharynx": (1.0, 0.71, 0.76),  # Light pink tissue
                "alveolar": (1.0, 0.84, 0.0),  # Golden yellow (nerve)
                "canal": (1.0, 0.84, 0.0),  # Golden yellow (nerve)
                "sinus": (0.68, 0.85, 0.9),  # Light blue (air)
                "maxillary_sinus": (0.68, 0.85, 0.9),  # Light blue (air)
                "default": (1.0, 1.0, 0.94)  # Default to enamel white
            }
        }
        # Icon file paths
        home_dir = os.path.expanduser("~")
        downloads_dir = os.path.join(home_dir, "Downloads")

        self.icon_paths = {
            "Nervous": os.path.join(downloads_dir, "brain_icon.png"),
            "Cardiovascular": os.path.join(downloads_dir, "heart_icon.png"),
            "Musculoskeletal": os.path.join(downloads_dir, "bone_icon.png"),
            "Dental": os.path.join(downloads_dir, "tooth_icon.png")
        }

        # Extraction parameters
        self.threshold = 0
        self.smoothing = 10

        # Cache directory
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".medical_viz_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Cache directory: {self.cache_dir}")

        self.init_ui()

    def init_ui(self):
        """Initialize the main UI"""
        self.setWindowTitle('Medical Visualization - NIfTI Auto-Loader')

        # Get screen size and set window to 70% of screen
        screen = QApplication.desktop().screenGeometry()
        width = int(screen.width() * 0.7)
        height = int(screen.height() * 0.7)
        x = int((screen.width() - width) / 2)
        y = int((screen.height() - height) / 2)

        self.setGeometry(x, y, width, height)
        self.setStyleSheet("background-color: #ffffff;")

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_widget.setLayout(main_layout)

        # Title bar
        title_bar = self.create_title_bar()
        main_layout.addWidget(title_bar)

        # Stacked widget for pages
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Create pages
        self.landing_page = self.create_landing_page()
        self.visualization_page = self.create_visualization_page()

        self.stacked_widget.addWidget(self.landing_page)
        self.stacked_widget.addWidget(self.visualization_page)

        # Show landing page
        self.stacked_widget.setCurrentWidget(self.landing_page)

        self.show()

    def create_title_bar(self):
        """Create top title bar"""
        bar = QFrame()
        bar.setFixedHeight(70)
        bar.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-bottom: 1px solid #e8e8e8;
            }
        """)

        layout = QHBoxLayout()
        layout.setContentsMargins(30, 0, 30, 0)
        bar.setLayout(layout)

        # Title
        title = QLabel("Medical Visualization")
        title.setFont(QFont("SF Pro Display", 20, QFont.Light))
        title.setStyleSheet("color: #1a1a1a; border: none;")
        layout.addWidget(title)

        layout.addStretch()

        # MRI Viewer button
        self.mri_viewer_btn = QPushButton("MRI Viewer")
        self.mri_viewer_btn.setFixedSize(120, 40)
        self.mri_viewer_btn.setStyleSheet("""
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
        """)
        self.mri_viewer_btn.setCursor(Qt.PointingHandCursor)
        self.mri_viewer_btn.clicked.connect(self.open_mri_viewer)
        layout.addWidget(self.mri_viewer_btn)

        layout.addSpacing(10)

        # Home button
        self.home_btn = QPushButton("←")
        self.home_btn.setFixedSize(40, 40)
        self.home_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #1a1a1a;
                border: 1px solid #e8e8e8;
                border-radius: 20px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
            }
        """)
        self.home_btn.setCursor(Qt.PointingHandCursor)
        self.home_btn.clicked.connect(self.go_home)
        self.home_btn.hide()
        layout.addWidget(self.home_btn)

        return bar

    def open_mri_viewer(self):
        """Open the MRI Viewer in a new window"""
        if self.mri_viewer_window is None or not self.mri_viewer_window.isVisible():
            self.mri_viewer_window = MRIViewer()
            self.mri_viewer_window.show()
        else:
            self.mri_viewer_window.raise_()
            self.mri_viewer_window.activateWindow()

    def create_landing_page(self):
        """Create the landing page with 4 system icons"""
        page = QWidget()
        page.setStyleSheet("background-color: #ffffff;")

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        page.setLayout(layout)

        layout.addSpacing(30)

        # Subtitle
        subtitle = QLabel("Select System")
        subtitle.setFont(QFont("SF Pro Display", 16, QFont.Light))
        subtitle.setStyleSheet("color: #666666;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(40)

        # Grid for 4 system cards
        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_widget.setLayout(grid_layout)
        grid_layout.setSpacing(30)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        systems = [
            ("Nervous", "Nervous", "#1a1a1a"),
            ("Cardiovascular", "Cardiovascular", "#1a1a1a"),
            ("Musculoskeletal", "Musculoskeletal", "#1a1a1a"),
            ("Dental", "Dental", "#1a1a1a")
        ]

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for (sys_key, sys_name, color), pos in zip(systems, positions):
            card = self.create_system_card(sys_key, sys_name, color)
            grid_layout.addWidget(card, pos[0], pos[1])

        layout.addWidget(grid_widget, alignment=Qt.AlignCenter)
        layout.addStretch()

        return page

    def create_system_card(self, sys_key, sys_name, accent_color):
        """Create a minimal clickable card for each system"""
        card = QFrame()
        card.setFixedSize(240, 240)
        card.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border: 1px solid #e8e8e8;
                border-radius: 12px;
            }
            QFrame:hover {
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
            }
        """)
        card.setCursor(Qt.PointingHandCursor)

        # Add shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 4)
        card.setGraphicsEffect(shadow)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        card.setLayout(layout)

        layout.addStretch()

        # Icon container
        icon_container = QFrame()
        icon_container.setFixedSize(100, 100)
        icon_container.setStyleSheet(f"""
            QFrame {{
                background-color: {accent_color};
                border-radius: 50px;
            }}
        """)

        # Icon image
        icon_label = QLabel(icon_container)
        icon_label.setFixedSize(100, 100)
        icon_label.setAlignment(Qt.AlignCenter)

        # Try to load icon
        icon_path = self.icon_paths.get(sys_key)
        icon_loaded = False

        if icon_path and os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                white_pixmap = QPixmap(scaled_pixmap.size())
                white_pixmap.fill(Qt.transparent)
                painter = QPainter(white_pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_Source)
                painter.drawPixmap(0, 0, scaled_pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                painter.fillRect(white_pixmap.rect(), QColor(255, 255, 255))
                painter.end()
                icon_label.setPixmap(white_pixmap)
                icon_loaded = True

        if not icon_loaded:
            # Fallback - show initial
            initials = {"Nervous": "N", "Cardiovascular": "C",
                        "Musculoskeletal": "M", "Dental": "D"}
            icon_label.setText(initials.get(sys_key, "?"))
            icon_label.setFont(QFont("SF Pro Display", 40, QFont.Bold))
            icon_label.setStyleSheet("color: white; background: transparent;")

        # Center icon
        icon_wrapper = QHBoxLayout()
        icon_wrapper.addStretch()
        icon_wrapper.addWidget(icon_container)
        icon_wrapper.addStretch()
        layout.addLayout(icon_wrapper)

        layout.addSpacing(20)

        # Title
        title_label = QLabel(sys_name)
        title_label.setFont(QFont("SF Pro Display", 16, QFont.Normal))
        title_label.setStyleSheet("color: #1a1a1a;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        layout.addStretch()

        # Make clickable
        card.mousePressEvent = lambda event: self.select_system(sys_key)

        return card

    def create_visualization_page(self):
        """Create the main visualization page"""
        page = QWidget()
        page.setStyleSheet("background-color: #ffffff;")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        page.setLayout(layout)

        # Left control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)

        # Right VTK viewer
        vtk_container = QFrame()
        vtk_container.setStyleSheet("background-color: #fafafa;")
        vtk_layout = QVBoxLayout()
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        vtk_container.setLayout(vtk_layout)

        self.vtk_widget = QVTKRenderWindowInteractor(vtk_container)
        vtk_layout.addWidget(self.vtk_widget)
        layout.addWidget(vtk_container)

        # Setup VTK
        self.setup_vtk()

        return page

    def create_control_panel(self):
        """Create the control panel"""
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-right: 1px solid #e8e8e8;
            }
        """)

        layout = QVBoxLayout()
        panel.setLayout(layout)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)

        # System info
        self.system_label = QLabel("System")
        self.system_label.setFont(QFont("SF Pro Display", 18, QFont.Medium))
        self.system_label.setStyleSheet("color: #1a1a1a;")
        layout.addWidget(self.system_label)

        # Status label
        self.status_label = QLabel("No data loaded")
        self.status_label.setFont(QFont("SF Pro Display", 11))
        self.status_label.setStyleSheet("color: #666666;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addSpacing(20)
       # --- CLIPPING PLANE CONTROLS ---
        self.clipping_controls = self.create_collapsible_group("Clipping Planes")
        clipping_layout = QVBoxLayout()

       # Define black slider style
        slider_style = """
    QSlider::groove:horizontal {
        height: 4px;
        background: #e8e8e8;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #1a1a1a;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }
    QSlider::handle:horizontal:hover {
        background: #333333;
    }
"""

       # Sagittal
        sag_label = QLabel("Sagittal (Left/Right)")
        sag_label.setStyleSheet("color: #666666; font-weight: 500;")
        clipping_layout.addWidget(sag_label)
        self.sagittal_check = QCheckBox("Enable")
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setRange(0, 100)
        self.sagittal_slider.setValue(50)
        self.sagittal_slider.setStyleSheet(slider_style)
        self.sagittal_label = QLabel("50%")
        self.sagittal_label.setStyleSheet("color: #999999; font-size: 11px;")
        self.sagittal_check.toggled.connect(self.update_clipping)
        self.sagittal_slider.valueChanged.connect(
    lambda v: (self.sagittal_label.setText(f"{v}%"), self.update_clipping()))
        clipping_layout.addWidget(self.sagittal_check)
        clipping_layout.addWidget(self.sagittal_slider)
        clipping_layout.addWidget(self.sagittal_label)

        clipping_layout.addSpacing(10)

       # Coronal
        cor_label = QLabel("Coronal (Front/Back)")
        cor_label.setStyleSheet("color: #666666; font-weight: 500;")
        clipping_layout.addWidget(cor_label)
        self.coronal_check = QCheckBox("Enable")
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setRange(0, 100)
        self.coronal_slider.setValue(50)
        self.coronal_slider.setStyleSheet(slider_style)
        self.coronal_label = QLabel("50%")
        self.coronal_label.setStyleSheet("color: #999999; font-size: 11px;")
        self.coronal_check.toggled.connect(self.update_clipping)
        self.coronal_slider.valueChanged.connect(
    lambda v: (self.coronal_label.setText(f"{v}%"), self.update_clipping()))
        clipping_layout.addWidget(self.coronal_check)
        clipping_layout.addWidget(self.coronal_slider)
        clipping_layout.addWidget(self.coronal_label)

        clipping_layout.addSpacing(10)

       # Axial
        ax_label = QLabel("Axial (Top/Bottom)")
        ax_label.setStyleSheet("color: #666666; font-weight: 500;")
        clipping_layout.addWidget(ax_label)
        self.axial_check = QCheckBox("Enable")
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setRange(0, 100)
        self.axial_slider.setValue(50)
        self.axial_slider.setStyleSheet(slider_style)
        self.axial_label = QLabel("50%")
        self.axial_label.setStyleSheet("color: #999999; font-size: 11px;")
        self.axial_check.toggled.connect(self.update_clipping)
        self.axial_slider.valueChanged.connect(
    lambda v: (self.axial_label.setText(f"{v}%"), self.update_clipping()))
        clipping_layout.addWidget(self.axial_check)
        clipping_layout.addWidget(self.axial_slider)
        clipping_layout.addWidget(self.axial_label)

        self.clipping_controls.content_widget.setLayout(clipping_layout)
        layout.addWidget(self.clipping_controls)
        self.clipping_controls.setVisible(False)  # Hide by default
       
        # Buttons
        reset_btn = self.create_minimal_button("Reset View")
        reset_btn.clicked.connect(self.reset_camera)
        layout.addWidget(reset_btn)

        # Animation button (only for Cardiovascular system)
        self.animation_btn = self.create_minimal_button("Show Animation")
        self.animation_btn.clicked.connect(self.toggle_animation)
        self.animation_btn.hide()
        layout.addWidget(self.animation_btn)

         # --- Add Clipping Planes toggle button ---
        self.clipping_toggle_btn = self.create_minimal_button("Show Clipping Planes")
        self.clipping_toggle_btn.clicked.connect(self.toggle_clipping_planes_ui)
        layout.addWidget(self.clipping_toggle_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e8e8e8;
                border-radius: 4px;
                text-align: center;
                background-color: #fafafa;
            }
            QProgressBar::chunk {
                background-color: #1a1a1a;
                border-radius: 3px;
            }
        """)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Fly-Through Section
        self.flythrough_section = self.create_flythrough_controls()
        layout.addWidget(self.flythrough_section)

        # Parts Section
        self.parts_scroll = self.create_parts_panel()
        parts_section = self.create_collapsible_section("Parts", self.parts_scroll)
        layout.addWidget(parts_section, stretch=1)

        layout.addStretch()

        return panel

    def create_flythrough_controls(self):
        """Create fly-through control section"""
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header = QLabel("🎥 Camera Fly-Through")
        header.setFont(QFont("SF Pro Display", 13, QFont.Bold))
        header.setStyleSheet("color: #1a1a1a;")
        layout.addWidget(header)

        # Define Path button
        self.define_path_btn = self.create_minimal_button("Define Path")
        self.define_path_btn.clicked.connect(self.toggle_path_definition)
        layout.addWidget(self.define_path_btn)

        # Path info label
        self.path_info_label = QLabel("No path defined")
        self.path_info_label.setFont(QFont("SF Pro Display", 10))
        self.path_info_label.setStyleSheet("color: #888888;")
        self.path_info_label.setWordWrap(True)
        layout.addWidget(self.path_info_label)

        # Play/Stop button
        self.play_fly_btn = self.create_minimal_button("▶ Play")
        self.play_fly_btn.clicked.connect(self.toggle_flythrough)
        self.play_fly_btn.setEnabled(False)
        layout.addWidget(self.play_fly_btn)

        # Speed slider
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet("color: #666666; font-size: 10px;")
        speed_layout.addWidget(speed_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 100)
        self.speed_slider.setValue(30)
        self.speed_slider.setStyleSheet("""
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
                margin: -5px 0;
                border-radius: 7px;
            }
        """)
        speed_layout.addWidget(self.speed_slider)
        layout.addLayout(speed_layout)

        # Clear Path button
        clear_path_btn = QPushButton("Clear Path")
        clear_path_btn.setFixedHeight(32)
        clear_path_btn.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #666666;
                border: 1px solid #e8e8e8;
                border-radius: 6px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
            }
        """)
        clear_path_btn.clicked.connect(self.clear_fly_path)
        layout.addWidget(clear_path_btn)

        return wrapper

    def create_collapsible_group(self, title):
        """Create a collapsible group box"""
        group = QFrame()
        group.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin: 2px;
            }
        """)

        main_layout = QVBoxLayout(group)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header button
        header = QPushButton(f"▼ {title}")
        header.setCheckable(True)
        header.setChecked(True)
        header.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                text-align: left;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:checked {
                background-color: #1565C0;
            }
        """)
        header.setCursor(Qt.PointingHandCursor)
        main_layout.addWidget(header)

        # Content widget
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: transparent; border: none;")
        main_layout.addWidget(content_widget)

        # Collapse/expand toggle
        def toggle():
            expanded = header.isChecked()
            header.setText(f"{'▼' if expanded else '▶'} {title}")
            content_widget.setVisible(expanded)

        header.clicked.connect(toggle)

        # Store content widget reference for later use
        group.content_widget = content_widget

        return group

    def create_collapsible_section(self, title, content_widget):
        """Creates a collapsible section with a toggle button."""
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(6)

        # Header button
        header_btn = QPushButton(f"▾ {title}")
        header_btn.setCheckable(True)
        header_btn.setChecked(True)
        header_btn.setCursor(Qt.PointingHandCursor)
        header_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                background-color: #f1f1f1;
                border: none;
                padding: 6px 10px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e9e9e9;
            }
        """)

        # Put the content_widget inside a container
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(content_widget)

        def toggle():
            expanded = header_btn.isChecked()
            header_btn.setText(f"{'▾' if expanded else '▸'} {title}")
            if expanded:
                content_container.setMaximumHeight(16777215)
                content_container.setVisible(True)
            else:
                content_container.setMaximumHeight(0)
                content_container.setVisible(False)

        header_btn.clicked.connect(toggle)

        wrapper_layout.addWidget(header_btn)
        wrapper_layout.addWidget(content_container)

        return wrapper

    def create_parts_panel(self):
        """Create the scrollable parts list panel"""
        self.parts_container = QWidget()
        self.parts_layout = QVBoxLayout(self.parts_container)
        self.parts_layout.setContentsMargins(0, 0, 0, 0)
        self.parts_layout.setSpacing(8)
        self.parts_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.parts_container)
        scroll.setStyleSheet("QScrollArea { border: none; background: #ffffff; }")

        return scroll

    def create_minimal_button(self, text):
        """Create minimal button"""
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
            QPushButton:disabled {
                background-color: #e8e8e8;
                color: #aaaaaa;
            }
        """)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def add_part_control(self, part_name, actor):
        """Add a part entry with color + opacity controls"""
        row = QFrame()
        row.setStyleSheet("background:#fafafa; border:1px solid #eaeaea; border-radius:6px;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 4, 8, 4)
        row_layout.setSpacing(8)

        # Part label
        label = QLabel(part_name)
        label.setStyleSheet("color:#222; font-weight:500;")
        label.setFixedWidth(80)
        row_layout.addWidget(label)

        # Focus button
        focus_btn = QPushButton("👁")
        focus_btn.setFixedSize(28, 28)
        focus_btn.setStyleSheet("""
            QPushButton {
                background-color: #e8e8e8;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #d0d0d0; }
            QPushButton:pressed { background-color: #b8b8b8; }
        """)
        focus_btn.setCursor(Qt.PointingHandCursor)
        focus_btn.clicked.connect(lambda: self.focus_on_actor(actor))
        row_layout.addWidget(focus_btn)

        # Color button
        color_btn = QPushButton()
        color_btn.setFixedSize(24, 24)
        color = actor.GetProperty().GetColor()
        color_btn.setStyleSheet(
            f"background-color: rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}); border-radius:4px;")
        color_btn.clicked.connect(lambda: self.change_actor_color(actor, color_btn))
        row_layout.addWidget(color_btn)

        # Opacity spinbox
        opacity_spin = QSpinBox()
        opacity_spin.setRange(0, 100)
        opacity_spin.setValue(int(actor.GetProperty().GetOpacity() * 100))
        opacity_spin.setFixedWidth(60)
        opacity_spin.valueChanged.connect(lambda val: self.change_actor_opacity(actor, val))
        row_layout.addWidget(opacity_spin)

        self.parts_layout.insertWidget(self.parts_layout.count() - 1, row)

    def change_actor_color(self, actor, button):
        """Open color picker and update actor color"""
        current_color = actor.GetProperty().GetColor()
        initial = QColor(int(current_color[0] * 255), int(current_color[1] * 255), int(current_color[2] * 255))
        color = QColorDialog.getColor(initial, self, "Choose Color")

        if color.isValid():
            rgb = (color.red() / 255, color.green() / 255, color.blue() / 255)
            actor.GetProperty().SetColor(rgb)
            button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border-radius:4px;")
            self.vtk_widget.GetRenderWindow().Render()

    def change_actor_opacity(self, actor, value):
        """Change actor opacity (0–100%)"""
        actor.GetProperty().SetOpacity(value / 100.0)
        self.vtk_widget.GetRenderWindow().Render()

    def focus_on_actor(self, focused_actor):
        """Focus camera on a specific actor and dim others"""
        if not hasattr(self, 'original_opacities'):
            self.original_opacities = {}

        bounds = focused_actor.GetBounds()
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]

        diagonal = ((bounds[1] - bounds[0]) ** 2 +
                    (bounds[3] - bounds[2]) ** 2 +
                    (bounds[5] - bounds[4]) ** 2) ** 0.5

        for actor in self.current_actors:
            if actor not in self.original_opacities:
                self.original_opacities[actor] = actor.GetProperty().GetOpacity()

            if actor == focused_actor:
                actor.GetProperty().SetOpacity(1.0)
            else:
                actor.GetProperty().SetOpacity(0.15)

        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetPosition(
            center[0] + diagonal * 0.8,
            center[1] + diagonal * 0.8,
            center[2] + diagonal * 0.8
        )
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def setup_vtk(self):
        """Setup VTK renderer with cinematic lighting and unrestricted camera clipping"""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)

        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Set custom interactor style for path picking
        self.custom_style = CustomInteractorStyle(parent=self)
        self.interactor.SetInteractorStyle(self.custom_style)

        self.interactor.Initialize()

        # Set camera clipping range to allow deep zoom
        camera = self.renderer.GetActiveCamera()
        camera.SetClippingRange(0.001, 10000.0)

        # Cinematic lighting
        light1 = vtk.vtkLight()
        light1.SetPosition(100, 100, 100)
        light1.SetIntensity(0.8)
        light1.SetColor(1.0, 1.0, 1.0)
        self.renderer.AddLight(light1)

        light2 = vtk.vtkLight()
        light2.SetPosition(-100, 50, 50)
        light2.SetIntensity(0.4)
        light2.SetColor(0.8, 0.9, 1.0)
        self.renderer.AddLight(light2)

        light3 = vtk.vtkLight()
        light3.SetPosition(0, -100, 50)
        light3.SetIntensity(0.3)
        light3.SetColor(0.6, 0.7, 0.9)
        self.renderer.AddLight(light3)

        # --- Initialize clipping plane visuals ---
        self.init_visual_clipping_planes()


    def init_visual_clipping_planes(self):
        """Initialize the visible geometric planes for clipping"""
        # Sagittal plane (X)
        self.plane_source_sagittal = vtk.vtkPlaneSource()
        self.plane_mapper_sagittal = vtk.vtkPolyDataMapper()
        self.plane_mapper_sagittal.SetInputConnection(self.plane_source_sagittal.GetOutputPort())
        self.plane_actor_sagittal = vtk.vtkActor()
        self.plane_actor_sagittal.SetMapper(self.plane_mapper_sagittal)
        self.plane_actor_sagittal.GetProperty().SetColor(1, 0, 0)  # red
        self.plane_actor_sagittal.GetProperty().SetOpacity(0.2)
        self.plane_actor_sagittal.SetVisibility(False)
        self.renderer.AddActor(self.plane_actor_sagittal)

        # Coronal plane (Y)
        self.plane_source_coronal = vtk.vtkPlaneSource()
        self.plane_mapper_coronal = vtk.vtkPolyDataMapper()
        self.plane_mapper_coronal.SetInputConnection(self.plane_source_coronal.GetOutputPort())
        self.plane_actor_coronal = vtk.vtkActor()
        self.plane_actor_coronal.SetMapper(self.plane_mapper_coronal)
        self.plane_actor_coronal.GetProperty().SetColor(0, 1, 0)  # green
        self.plane_actor_coronal.GetProperty().SetOpacity(0.2)
        self.plane_actor_coronal.SetVisibility(False)
        self.renderer.AddActor(self.plane_actor_coronal)

        # Axial plane (Z)
        self.plane_source_axial = vtk.vtkPlaneSource()
        self.plane_mapper_axial = vtk.vtkPolyDataMapper()
        self.plane_mapper_axial.SetInputConnection(self.plane_source_axial.GetOutputPort())
        self.plane_actor_axial = vtk.vtkActor()
        self.plane_actor_axial.SetMapper(self.plane_mapper_axial)
        self.plane_actor_axial.GetProperty().SetColor(0, 0, 1)  # blue
        self.plane_actor_axial.GetProperty().SetOpacity(0.2)
        self.plane_actor_axial.SetVisibility(False)
        self.renderer.AddActor(self.plane_actor_axial)

    def toggle_path_definition(self):
        """Toggle path definition mode"""
        if self.path_definition_mode:
            # Exit path definition mode
            self.path_definition_mode = False
            self.define_path_btn.setText("Define Path")
            self.status_label.setText("Path definition completed")

            # Enable play button if we have points
            if len(self.fly_path_points) >= 2:
                self.play_fly_btn.setEnabled(True)
                self.path_info_label.setText(f"Path with {len(self.fly_path_points)} points")
            else:
                self.path_info_label.setText("Need at least 2 points")
        else:
            # Enter path definition mode
            self.path_definition_mode = True
            self.define_path_btn.setText("Done")
            self.status_label.setText("Click on the model to define camera path")
            self.path_info_label.setText("Adding points...")

    def update_path_visual(self):
        """Update the visual representation of the fly-through path"""
        # Remove old path actor
        if self.path_actor:
            self.renderer.RemoveActor(self.path_actor)
            self.path_actor = None

        # Create new path visualization
        if len(self.fly_path_points) > 0:
            self.path_actor = PathVisualizer.create_path_actor(
                self.fly_path_points,
                self.renderer
            )
            if self.path_actor:
                self.renderer.AddActor(self.path_actor)

    def clear_fly_path(self):
        """Clear the fly-through path"""
        self.fly_path_points.clear()

        # Remove path visualization
        if self.path_actor:
            self.renderer.RemoveActor(self.path_actor)
            self.path_actor = None

        # Reset UI
        self.path_info_label.setText("No path defined")
        self.play_fly_btn.setEnabled(False)

        # Stop flythrough if running
        if self.is_flying:
            self.stop_flythrough()

        self.vtk_widget.GetRenderWindow().Render()

    def toggle_flythrough(self):
        """Toggle fly-through animation"""
        if self.is_flying:
            self.stop_flythrough()
        else:
            self.start_flythrough()

    def start_flythrough(self):
        """Start the camera fly-through animation"""
        if len(self.fly_path_points) < 2:
            QMessageBox.warning(self, "No Path", "Please define a camera path first")
            return

        # Create camera animator
        camera = self.renderer.GetActiveCamera()
        self.camera_animator = CameraAnimator(camera, easing_type='ease_in_out')

        # Set path with interpolation
        steps_per_segment = max(10, 100 // len(self.fly_path_points))
        self.camera_animator.set_path(self.fly_path_points, steps_per_segment)

        # Hide path visualization during flythrough
        if self.path_actor:
            self.path_actor.SetVisibility(False)

        # Start animation timer
        if self.flythrough_timer is None:
            self.flythrough_timer = QTimer(self)
            self.flythrough_timer.timeout.connect(self.update_flythrough_frame)

        fps = self.speed_slider.value()
        self.flythrough_timer.start(1000 // fps)

        self.is_flying = True
        self.play_fly_btn.setText("⏸ Stop")
        self.status_label.setText("🎬 Flying through...")

    def update_flythrough_frame(self):
        """Update one frame of the fly-through animation"""
        if not self.camera_animator:
            return

        # Advance camera along path
        still_moving = self.camera_animator.step(loop=True)

        if not still_moving:
            self.stop_flythrough()
            return

        # Update clipping range to allow deep zoom
        camera = self.renderer.GetActiveCamera()
        camera.SetClippingRange(0.001, 10000.0)

        # Render
        self.vtk_widget.GetRenderWindow().Render()

    def stop_flythrough(self):
        """Stop the fly-through animation"""
        if self.flythrough_timer:
            self.flythrough_timer.stop()

        # Show path visualization again
        if self.path_actor:
            self.path_actor.SetVisibility(True)

        self.is_flying = False
        self.play_fly_btn.setText("▶ Play")
        self.status_label.setText("Fly-through stopped")

        self.vtk_widget.GetRenderWindow().Render()

    def toggle_clipping(self, state):
        """Toggle clipping plane on/off"""
        if state and self.is_flying:
            self.clipping_manager.enable(self.current_actors)
        else:
            self.clipping_manager.disable()

        if not self.is_flying:
            self.vtk_widget.GetRenderWindow().Render()

    # ========== SYSTEM LOADING METHODS ==========

    def select_system(self, system_name):
        """Select a system and switch to visualization page"""
        # Clear old UI entries
        for i in reversed(range(self.parts_layout.count())):
            widget = self.parts_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.current_system = system_name
        self.system_label.setText(system_name)
        self.stacked_widget.setCurrentWidget(self.visualization_page)
        self.home_btn.show()

        # Show animation button only for Cardiovascular system
        if system_name == "Cardiovascular":
            self.animation_btn.show()
        else:
            self.animation_btn.hide()

        # Clear fly-through path when switching systems
        self.clear_fly_path()

        # Automatically load the system
        self.load_system(system_name)

    def go_home(self):
        """Return to landing page"""
        if self.is_animating:
            self.stop_animation()

        if self.is_flying:
            self.stop_flythrough()

        self.stacked_widget.setCurrentWidget(self.landing_page)
        self.home_btn.hide()
        self.clear_all_actors()
        self.clear_fly_path()
        self.vtk_widget.GetRenderWindow().Render()

    def clear_all_actors(self):
        """Remove all actors from the scene"""
        for actor in self.current_actors:
            self.renderer.RemoveActor(actor)
        self.current_actors.clear()

    def get_structure_color(self, filename):
        """Get color for structure based on filename"""
        if not self.current_system:
            return (0.7, 0.7, 0.7)

        colors = self.color_schemes.get(self.current_system, {})
        filename_lower = filename.lower()

        # Special handling for Dental system with priority matching
        if self.current_system == "Dental":
            # Check for pulp first (highest priority)
            if "pulp" in filename_lower:
                return colors.get("pulp", (0.86, 0.08, 0.24))

            # Check for jawbone
            if "jawbone" in filename_lower:
                return colors.get("jawbone", (0.96, 0.96, 0.86))

            # Check for specific tooth types
            if "molar" in filename_lower:
                return colors.get("molar", (1.0, 1.0, 0.94))
            if "incisor" in filename_lower:
                return colors.get("incisor", (1.0, 1.0, 0.94))
            if "canine" in filename_lower:
                return colors.get("canine", (1.0, 1.0, 0.94))
            if "premolar" in filename_lower:
                return colors.get("premolar", (1.0, 1.0, 0.94))

            # Check for other dental structures
            if "pharynx" in filename_lower:
                return colors.get("pharynx", (1.0, 0.71, 0.76))
            if "alveolar" in filename_lower or "canal" in filename_lower:
                return colors.get("alveolar", (1.0, 0.84, 0.0))
            if "sinus" in filename_lower or "maxillary_sinus" in filename_lower:
                return colors.get("sinus", (0.68, 0.85, 0.9))

            # Generic tooth (if contains "fdi" number but no specific type matched)
            if "fdi" in filename_lower:
                return colors.get("tooth", (1.0, 1.0, 0.94))

            return colors.get("default", (1.0, 1.0, 0.94))

        # For other systems, use original keyword matching
        for keyword, color in colors.items():
            if keyword in filename_lower:
                return color

        return colors.get("default", (0.7, 0.7, 0.7))

    def get_file_hash(self, filepath):
        """Generate a hash for the file to use as cache key"""
        stat = os.stat(filepath)
        hash_string = f"{filepath}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def get_cache_path(self, file_hash):
        """Get the cache file path for a given hash"""
        return os.path.join(self.cache_dir, f"{file_hash}.pkl")

    def get_stl_cache_path(self, file_hash):
        """Get the cache file path for STL files"""
        return os.path.join(self.cache_dir, f"stl_{file_hash}.pkl")

    def save_stl_to_cache(self, filepath, polydata, color):
        """Save STL as compressed NumPy arrays (10x faster than VTK format)"""
        try:
            file_hash = self.get_file_hash(filepath)
            cache_path = self.get_stl_cache_path(file_hash)
            npz_path = cache_path.replace('.pkl', '.npz')

            # Extract vertices and faces as NumPy arrays
            points = polydata.GetPoints()
            num_points = points.GetNumberOfPoints()
            verts = np.array([points.GetPoint(i) for i in range(num_points)])

            # Extract faces
            polys = polydata.GetPolys()
            polys.InitTraversal()
            faces = []
            idList = vtk.vtkIdList()
            while polys.GetNextCell(idList):
                faces.append([idList.GetId(j) for j in range(idList.GetNumberOfIds())])
            faces = np.array(faces)

            # Save as compressed NumPy (MUCH faster to load than VTK or pickle)
            np.savez_compressed(npz_path,
                                vertices=verts,
                                faces=faces,
                                color=np.array(color))

            return True
        except Exception as e:
            print(f"Failed to cache STL {filepath}: {e}")
            return False

    def load_stl_from_cache(self, filepath):
        """Load STL from compressed NumPy cache (10x faster)"""
        try:
            file_hash = self.get_file_hash(filepath)
            cache_path = self.get_stl_cache_path(file_hash)
            npz_path = cache_path.replace('.pkl', '.npz')

            if not os.path.exists(npz_path):
                return None

            # Load from compressed NumPy
            data = np.load(npz_path)
            verts = data['vertices']
            faces = data['faces']
            color = tuple(data['color'])

            # Convert back to VTK polydata
            polydata = self.numpy_to_polydata(verts, faces)

            return {
                'polydata': polydata,
                'color': color,
                'filename': os.path.basename(filepath)
            }

        except Exception as e:
            print(f"Failed to load STL cache: {e}")
            return None

    def numpy_to_polydata(self, verts, faces):
        """Fast conversion from NumPy arrays to VTK polydata"""
        # Create points
        points = vtk.vtkPoints()
        for vert in verts:
            points.InsertNextPoint(vert[0], vert[1], vert[2])

        # Create cell array for faces
        cells = vtk.vtkCellArray()
        for face in faces:
            cells.InsertNextCell(len(face))
            for vertex_id in face:
                cells.InsertCellPoint(int(vertex_id))

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        return polydata

    def save_to_cache(self, filepath, verts, faces, color, filename):
        """Save extracted mesh data to cache"""
        try:
            file_hash = self.get_file_hash(filepath)
            cache_path = self.get_cache_path(file_hash)

            cache_data = {
                'verts': verts,
                'faces': faces,
                'color': color,
                'filename': filename,
                'threshold': self.threshold,
                'smoothing': self.smoothing
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"✓ Cached: {filename}")
            return True
        except Exception as e:
            print(f"Failed to cache {filename}: {e}")
            return False

    def load_from_cache(self, filepath):
        """Load mesh data from cache if available"""
        try:
            file_hash = self.get_file_hash(filepath)
            cache_path = self.get_cache_path(file_hash)

            if not os.path.exists(cache_path):
                return None

            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Verify cache is for same settings
            if (cache_data.get('threshold') == self.threshold and
                    cache_data.get('smoothing') == self.smoothing):
                print(f"✓ Loaded from cache: {cache_data['filename']}")
                return cache_data
            else:
                print(f"Cache outdated for {cache_data['filename']}")
                return None

        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None

    def load_system(self, system_name):
        """Load all NIfTI files from the system folder automatically"""
        if not HAS_NIBABEL:
            QMessageBox.critical(self, "Missing Libraries",
                                 "Required libraries not installed.\n\n"
                                 "Please run:\npip install nibabel scikit-image numpy scipy")
            return

        # Clear existing actors
        self.clear_all_actors()

        # Get folder path
        folder_path = self.organ_folders.get(system_name)

        if not folder_path or not os.path.exists(folder_path):
            self.status_label.setText(f"Folder not found:\n{folder_path}")
            QMessageBox.warning(self, "Folder Not Found",
                                f"The folder does not exist:\n{folder_path}\n\n"
                                "Please update the path in the code.")
            return

        # Find all NIfTI files
        nifti_files = glob.glob(os.path.join(folder_path, "*.nii.gz"))
        nifti_files += glob.glob(os.path.join(folder_path, "*.nii"))

        if not nifti_files:
            self.status_label.setText(f"No NIfTI files found in:\n{folder_path}")
            QMessageBox.warning(self, "No Files",
                                f"No .nii or .nii.gz files found in:\n{folder_path}")
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(nifti_files))
        self.progress_bar.setValue(0)

        self.status_label.setText(f"⏳ Loading {len(nifti_files)} file(s)...")
        QApplication.processEvents()

        # Load each file
        loaded_count = 0
        for i, filepath in enumerate(nifti_files):
            self.progress_bar.setValue(i)
            QApplication.processEvents()

            if self.load_single_nifti(filepath, self.threshold, self.smoothing):
                loaded_count += 1

        self.progress_bar.setValue(len(nifti_files))
        self.progress_bar.setVisible(False)

        if loaded_count > 0:
            self.renderer.ResetCamera()
            # Set extended clipping range for deep zoom
            camera = self.renderer.GetActiveCamera()
            camera.SetClippingRange(0.001, 10000.0)
            self.vtk_widget.GetRenderWindow().Render()

            # --- Center clipping planes at the middle of the whole body ---
            self.calculate_bounds()
            if self.bounds:
                x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                cz = (z_min + z_max) / 2

                # Save global center for clipping
                self.body_center = (cx, cy, cz)

                # Reset sliders to middle position
                self.sagittal_slider.setValue(50)
                self.coronal_slider.setValue(50)
                self.axial_slider.setValue(50)

                # Ensure planes start centered
                self.clip_planes_enabled = True
                self.update_clipping()

            self.status_label.setText(f"Loaded {loaded_count}/{len(nifti_files)} structures\n\n"
                                      f"• Rotate: Left-click drag\n"
                                      f"• Zoom: Right-click drag\n"
                                      f"• Pan: Middle-click drag")
        else:
            self.status_label.setText(f"Failed to load structures")

    def load_single_nifti(self, filepath, threshold, smoothing):
        """Load and extract surface from a single NIfTI file using Marching Cubes WITH CACHING"""
        try:
            filename = Path(filepath).stem

            # Try to load from cache first
            cache_data = self.load_from_cache(filepath)

            if cache_data is not None:
                verts = cache_data['verts']
                faces = cache_data['faces']
                color = cache_data['color']

                # Create VTK actor from cached data
                actor = self.create_vtk_actor(verts, faces, smoothing)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().SetInterpolationToPhong()
                actor.GetProperty().SetSpecular(0.3)
                actor.GetProperty().SetSpecularPower(30)



                # --- ADDED: Set actor to use the clipping plane collection ---
                actor.GetMapper().SetClippingPlanes(self.clip_plane_collection)

                self.renderer.AddActor(actor)
                self.current_actors.append(actor)
                self.add_part_control(filename, actor)

                return True

            # Cache miss - process the NIfTI file
            print(f"Processing (not cached): {filename}")

            nii = nib.load(filepath)
            data = nii.get_fdata()
            spacing = nii.header.get_zooms()[:3]

            if data.max() <= data.min():
                print(f"Skipping {filepath}: no variation in data")
                return False

            # Normalize data to 0-255
            data_norm = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

            # MARCHING CUBES - Extract surface mesh
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    data_norm,
                    level=threshold,
                    spacing=spacing
                )
            except Exception as e:
                print(f"Marching cubes failed for {filepath}: {e}")
                return False

            if len(verts) == 0:
                print(f"No vertices extracted from {filepath}")
                return False

            # Get color for this structure
            color = self.get_structure_color(filename)

            # Save to cache for next time
            self.save_to_cache(filepath, verts, faces, color, filename)

            # Create VTK actor
            actor = self.create_vtk_actor(verts, faces, smoothing)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetInterpolationToPhong()
            actor.GetProperty().SetSpecular(0.3)
            actor.GetProperty().SetSpecularPower(30)

            # Add to scene
            self.renderer.AddActor(actor)
            self.current_actors.append(actor)
            self.add_part_control(filename, actor)

            print(f"✓ Processed and loaded: {filename} ({len(verts)} vertices)")
            return True

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return False

    def create_vtk_actor(self, verts, faces, smoothing):
        """Create VTK actor from mesh vertices and faces"""
        # Create VTK points from vertices
        points = vtk.vtkPoints()
        for v in verts:
            points.InsertNextPoint(*v)

        # Create triangles from faces
        triangles = vtk.vtkCellArray()
        for f in faces:
            triangle = vtk.vtkTriangle()
            for j in range(3):
                triangle.GetPointIds().SetId(j, int(f[j]))
            triangles.InsertNextCell(triangle)

        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(triangles)

        # Apply smoothing if requested
        if smoothing > 0:
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputData(polydata)
            smoother.SetNumberOfIterations(smoothing)
            smoother.Update()
            polydata = smoother.GetOutput()

        # Compute normals for proper lighting
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.Update()

        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())

        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def reset_camera(self):
        """Reset camera view and focus"""
        if hasattr(self, 'original_opacities'):
            for actor, original_opacity in self.original_opacities.items():
                actor.GetProperty().SetOpacity(original_opacity)
            self.original_opacities.clear()
        else:
            for actor in self.current_actors:
                actor.GetProperty().SetOpacity(1.0)

        self.renderer.ResetCamera()
        # Set extended clipping range for deep zoom capability
        camera = self.renderer.GetActiveCamera()
        camera.SetClippingRange(0.001, 10000.0)
        self.vtk_widget.GetRenderWindow().Render()

    # ========== ANIMATION METHODS ==========

    # ========== SYSTEM LOADING METHODS ==========

    def toggle_animation(self):
        """Toggle heart beat animation on/off"""
        if self.is_animating:
            self.stop_animation()
        else:
            self.start_animation()

    def start_animation(self):
        """Start animation with PARALLEL loading for 10x speed improvement"""
        try:
            if not os.path.exists(self.heart_animation_folder):
                QMessageBox.warning(self, "Animation Not Found",
                                    f"Animation folder not found:\n{self.heart_animation_folder}")
                return

            stl_files = sorted(glob.glob(os.path.join(self.heart_animation_folder, "*.stl")))
            if not stl_files:
                QMessageBox.warning(self, "No Animation Files",
                                    f"No STL files found in:\n{self.heart_animation_folder}")
                return

            # Group by frame
            frame_pattern = re.compile(r'frame_(\d+)_(.+)\.stl')
            frames_dict = {}
            for filepath in stl_files:
                filename = os.path.basename(filepath)
                match = frame_pattern.match(filename)
                if match:
                    frame_num = int(match.group(1))
                    if frame_num not in frames_dict:
                        frames_dict[frame_num] = []
                    frames_dict[frame_num].append(filepath)
                else:
                    frame_idx = len(frames_dict)
                    frames_dict[frame_idx] = [filepath]

            sorted_frames = sorted(frames_dict.keys())
            animation_file_paths = [frames_dict[frame] for frame in sorted_frames]
            self.total_frames = len(animation_file_paths)

            self.status_label.setText(f"⚡ Loading {len(stl_files)} files with parallel processing...")
            QApplication.processEvents()

            # Hide static actors
            for actor in self.current_actors:
                actor.SetVisibility(False)

            self.preloaded_frames.clear()
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(stl_files))

            # PARALLEL LOADING with ThreadPoolExecutor (4-8x faster!)
            file_counter = 0
            with ThreadPoolExecutor(max_workers=8) as executor:
                for frame_files in animation_file_paths:
                    # Submit all files in this frame to parallel loading
                    future_to_file = {
                        executor.submit(self.load_stl_file, filepath): filepath
                        for filepath in frame_files
                    }

                    frame_actors = []
                    for future in as_completed(future_to_file):
                        file_counter += 1
                        self.progress_bar.setValue(file_counter)

                        if file_counter % 50 == 0:  # Update UI every 50 files
                            QApplication.processEvents()

                        actor = future.result()
                        if actor:
                            actor.SetVisibility(False)
                            self.renderer.AddActor(actor)
                            frame_actors.append(actor)

                    self.preloaded_frames.append(frame_actors)

            self.progress_bar.setVisible(False)
            self.status_label.setText(f"▶ Animation ready ({self.total_frames} frames)")

            # Start animation
            if self.animation_timer is None:
                self.animation_timer = QTimer(self)
                self.animation_timer.timeout.connect(self.update_animation_frame)

            self.current_frame = -1
            self.is_animating = True
            self.animation_timer.start(33)  # 30 FPS
            self.animation_btn.setText("Stop Animation")

            print(f"✓ Animation loaded with parallel processing")

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_label.setText("❌ Animation load failed")
            QMessageBox.critical(self, "Error", f"Failed to load animation:\n{str(e)}")
            print(f"Error: {e}")
            traceback.print_exc()

    def update_animation_frame(self):
        """Update animation frame (called by timer)"""
        try:
            if not self.is_animating or not self.preloaded_frames:
                return

            # Hide current frame
            if 0 <= self.current_frame < len(self.preloaded_frames):
                for actor in self.preloaded_frames[self.current_frame]:
                    actor.SetVisibility(False)

            # Advance to next frame (loop)
            self.current_frame = (self.current_frame + 1) % len(self.preloaded_frames)

            # Show next frame
            for actor in self.preloaded_frames[self.current_frame]:
                actor.SetVisibility(True)

            # Render
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            print(f"Error in animation frame update: {e}")
            traceback.print_exc()
            self.stop_animation()

    def stop_animation(self):
        """Stop the heart beat animation and restore static scene"""
        try:
            if self.animation_timer:
                self.animation_timer.stop()

            # Hide and remove all preloaded frame actors
            for frame_actors in self.preloaded_frames:
                for actor in frame_actors:
                    actor.SetVisibility(False)
                    try:
                        self.renderer.RemoveActor(actor)
                    except Exception:
                        pass

            self.preloaded_frames.clear()
            self.total_frames = 0
            self.current_frame = 0

            # Restore static models visibility
            for actor in self.current_actors:
                actor.SetVisibility(True)

            self.is_animating = False
            self.animation_btn.setText("Show Animation")
            self.status_label.setText("⏹ Animation stopped")
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            print(f"Error stopping animation: {e}")
            traceback.print_exc()

    def on_viz_mode_changed(self, mode, checked):
        """Handle visualization mode change"""
        if not checked:
            return

        self.status_label.setText(f"Visualization: {mode}")

        # --- MODIFIED: Show/Hide clipping controls based on mode ---
        # Check if attribute exists, as this can be called during init
        if hasattr(self, 'clipping_controls'):
            if mode == "Clipping":
                self.clipping_controls.setVisible(True)
                self.enable_clipping()
            else:
                self.clipping_controls.setVisible(False)
                self.disable_clipping()
        # --- END MODIFIED ---

    def enable_clipping(self):
        """Enable clipping planes"""
        # --- FIX: Add check for controls ---
        if not hasattr(self, 'clipping_controls'):
            return  # Still initializing

        if not self.current_actors:
            # Only show warning if not initializing
            if hasattr(self, 'vtk_widget'):
                QMessageBox.warning(self, "No Data", "Please load anatomical data first.")
            return

        self.clip_planes_enabled = True
        self.clipping_controls.setVisible(True)
        self.update_clipping()  # Call to show and position planes
        self.status_label.setText("✂️ Clipping planes enabled")

    def disable_clipping(self):
        """Disable clipping planes"""
        self.clip_planes_enabled = False

        # --- FIX: Add check for controls ---
        if hasattr(self, 'clipping_controls'):
            self.clipping_controls.setVisible(False)

        # Remove all clipping planes from actors
        self.clip_plane_collection.RemoveAllItems()

        for actor in self.current_actors:
            actor.GetMapper().RemoveAllClippingPlanes()

        # --- Hide visual planes ---
        if self.plane_actor_sagittal:
            self.plane_actor_sagittal.SetVisibility(False)
            self.plane_actor_coronal.SetVisibility(False)
            self.plane_actor_axial.SetVisibility(False)

        # --- FIX: Check if vtk_widget exists before rendering ---
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()
            self.status_label.setText("Clipping planes disabled")
        else:
            # It's just initializing, no need to render or set status
            pass

    # --- END OF FIX ---

    def toggle_clipping_planes_ui(self):
        """Show or hide the clipping planes control UI"""
        if self.clipping_controls.isVisible():
            self.clipping_controls.setVisible(False)
            self.disable_clipping()
            self.clipping_toggle_btn.setText("Show Clipping Planes")
        else:
            self.enable_clipping()
            self.clipping_controls.setVisible(True)
            self.clipping_toggle_btn.setText("Hide Clipping Planes")

    def calculate_bounds(self):
        """Calculate overall bounds of all current actors"""
        if not self.current_actors:
            return None

        combined_bounds = [float('inf'), float('-inf'),
                           float('inf'), float('-inf'),
                           float('inf'), float('-inf')]

        for actor in self.current_actors:
            b = actor.GetBounds()
            combined_bounds[0] = min(combined_bounds[0], b[0])
            combined_bounds[1] = max(combined_bounds[1], b[1])
            combined_bounds[2] = min(combined_bounds[2], b[2])
            combined_bounds[3] = max(combined_bounds[3], b[3])
            combined_bounds[4] = min(combined_bounds[4], b[4])
            combined_bounds[5] = max(combined_bounds[5], b[5])

        self.bounds = combined_bounds
        return combined_bounds

    def update_clipping(self):
        """Update clipping plane positions AND visual plane actors"""
        if not self.clip_planes_enabled:
            return

        if not self.bounds:
            self.calculate_bounds()
            if not self.bounds:  # Still no bounds
                return

        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds

        # Use precomputed body center if available
        if hasattr(self, 'body_center'):
            cx, cy, cz = self.body_center
        else:
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            cz = (z_min + z_max) / 2

        # Clear collection
        self.clip_plane_collection.RemoveAllItems()

        # --- Define plane size (make them 1.5x the size of the model) ---
        size_mult = 1.5
        center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        x_size = (x_max - x_min) * size_mult
        y_size = (y_max - y_min) * size_mult
        z_size = (z_max - z_min) * size_mult

        # Handle zero-size dimensions
        if x_size == 0: x_size = max(y_size, z_size, 100)
        if y_size == 0: y_size = max(x_size, z_size, 100)
        if z_size == 0: z_size = max(x_size, y_size, 100)

        cx, cy, cz = center
        hsx, hsy, hsz = x_size / 2, y_size / 2, z_size / 2

        # Sagittal (X-axis)
        sag_enabled = self.sagittal_check.isChecked()
        self.plane_actor_sagittal.SetVisibility(sag_enabled)  # Show/hide visual plane
        if sag_enabled:
            pos = self.sagittal_slider.value() / 100.0
            origin_x = x_min + pos * (x_max - x_min)
            clip_origin = [origin_x, cy, cz]

            # Update clipping plane (the effect)
            self.clip_plane_sagittal.SetOrigin(clip_origin)
            self.clip_plane_sagittal.SetNormal(1, 0, 0)
            self.clip_plane_collection.AddItem(self.clip_plane_sagittal)

            # Update VISUAL plane (the 3D object)
            p_o = [origin_x, cy - hsy, cz - hsz]  # Origin
            p_1 = [origin_x, cy + hsy, cz - hsz]  # Point 1 (along Y)
            p_2 = [origin_x, cy - hsy, cz + hsz]  # Point 2 (along Z)
            self.plane_source_sagittal.SetOrigin(p_o)
            self.plane_source_sagittal.SetPoint1(p_1)
            self.plane_source_sagittal.SetPoint2(p_2)
            self.plane_source_sagittal.Update()

        # Coronal (Y-axis)
        cor_enabled = self.coronal_check.isChecked()
        self.plane_actor_coronal.SetVisibility(cor_enabled)
        if cor_enabled:
            pos = self.coronal_slider.value() / 100.0
            origin_y = y_min + pos * (y_max - y_min)
            clip_origin = [cx, origin_y, cz]

            self.clip_plane_coronal.SetOrigin(clip_origin)
            self.clip_plane_coronal.SetNormal(0, 1, 0)
            self.clip_plane_collection.AddItem(self.clip_plane_coronal)

            p_o = [cx - hsx, origin_y, cz - hsz]  # Origin
            p_1 = [cx + hsx, origin_y, cz - hsz]  # Point 1 (along X)
            p_2 = [cx - hsx, origin_y, cz + hsz]  # Point 2 (along Z)
            self.plane_source_coronal.SetOrigin(p_o)
            self.plane_source_coronal.SetPoint1(p_1)
            self.plane_source_coronal.SetPoint2(p_2)
            self.plane_source_coronal.Update()

        # Axial (Z-axis)
        ax_enabled = self.axial_check.isChecked()
        self.plane_actor_axial.SetVisibility(ax_enabled)
        if ax_enabled:
            pos = self.axial_slider.value() / 100.0
            origin_z = z_min + pos * (z_max - z_min)
            clip_origin = [cx, cy, origin_z]

            self.clip_plane_axial.SetOrigin(clip_origin)
            self.clip_plane_axial.SetNormal(0, 0, 1)
            self.clip_plane_collection.AddItem(self.clip_plane_axial)

            p_o = [cx - hsx, cy - hsy, origin_z]  # Origin
            p_1 = [cx + hsx, cy - hsy, origin_z]  # Point 1 (along X)
            p_2 = [cx - hsx, cy + hsy, origin_z]  # Point 2 (along Y)
            self.plane_source_axial.SetOrigin(p_o)
            self.plane_source_axial.SetPoint1(p_1)
            self.plane_source_axial.SetPoint2(p_2)
            self.plane_source_axial.Update()

        # Apply to all actors
        for actor in self.current_actors:
            actor.GetMapper().SetClippingPlanes(self.clip_plane_collection)

        self.vtk_widget.GetRenderWindow().Render()

    def load_stl_file(self, filepath):
        """Load a single STL file and return a vtkActor with proper coloring (WITH CACHING)"""
        try:
            if not os.path.exists(filepath):
                print(f"STL file not found: {filepath}")
                return None

            # TRY CACHE FIRST (FAST!)
            cache_data = self.load_stl_from_cache(filepath)

            if cache_data is not None:
                # Load from cache - MUCH FASTER!
                polydata = cache_data['polydata']
                color = cache_data['color']

                # Apply normals
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputData(polydata)
                normals.ComputePointNormalsOn()
                normals.Update()

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(normals.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().SetInterpolationToPhong()
                actor.GetProperty().SetSpecular(0.3)
                actor.GetProperty().SetSpecularPower(30)

                return actor

            # CACHE MISS - Load from STL file
            reader = vtk.vtkSTLReader()
            reader.SetFileName(filepath)
            reader.Update()
            polydata = reader.GetOutput()

            if polydata is None or polydata.GetNumberOfPoints() == 0:
                print(f"No points in STL: {filepath}")
                return None

            # Apply smoothing
            if self.smoothing > 0:
                smoother = vtk.vtkSmoothPolyDataFilter()
                smoother.SetInputData(polydata)
                smoother.SetNumberOfIterations(self.smoothing)
                smoother.Update()
                polydata = smoother.GetOutput()

            # GET COLOR BASED ON FILENAME
            filename = Path(filepath).stem
            color = self.get_structure_color(filename)

            # SAVE TO CACHE for next time (in background, don't block)
            try:
                self.save_stl_to_cache(filepath, polydata, color)
            except:
                pass  # Don't fail if caching fails

            # Create actor
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.ComputePointNormalsOn()
            normals.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetInterpolationToPhong()
            actor.GetProperty().SetSpecular(0.3)
            actor.GetProperty().SetSpecularPower(30)

            return actor

        except Exception as e:
            print(f"Error loading STL {filepath}: {e}")
            traceback.print_exc()
            return None






def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("SF Pro Display", 10))
    window = MedicalVisualization()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()