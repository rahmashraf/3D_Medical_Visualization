# Medical Visualization System 
**A Medical imaging visualization application built with Python, VTK, and PyQt5 for interactive 3D visualization and Multi-Planar Reconstruction (MPR) of anatomical structures from NIfTI and STL files.**

___

# Overview
This comprehensive medical visualization system provides two main components:

- 3D Anatomical Viewer: Interactive 3D visualization of anatomical structures with advanced features like clipping planes, camera fly-through, and animation support.
-  MRI/MPR Viewer: Multi-planar reconstruction viewer with support for axial, coronal, sagittal views, and curved/panoramic MPR.
  
---

# Features
3D Visualization 
Core Functionality

- **Multi-System Support:**

![interface](https://github.com/rahmashraf/3D_Medical_Visualization/blob/main/assets/main.png)

- **Surface Rendering:**


- **MRI Viewer (Curved MPR)**
- Core Views

Axial View (X-Y plane): Top-down brain slices
Coronal View (X-Z plane): Front-to-back slices
Sagittal View (Y-Z plane): Side view slices
**Curved MPR View**: Custom curved reconstructions

- MPR Features

Interactive Crosshairs: Synchronized across all views
Curved Planar MPR:

Click-to-define curved path on axial view
Z-slicing along the curve
Real-time interpolation
- Display Options:

7 colormap choices (gray, viridis, plasma, etc.)
Aspect ratio correction
Real-time voxel value display
Full-height curved reconstruction
Dental arch visualization

![CurvedMpr](https://github.com/rahmashraf/3D_Medical_Visualization/blob/main/assets/curvedmpr.gif)

- Automatic NIfTI Processing:

Batch loading of .nii and .nii.gz files
Marching Cubes surface extraction
Smart color coding by anatomical structure


- Heart Animation:

Real-time cardiac cycle visualization
STL frame sequence support
30 FPS playback with parallel loading


- Camera Fly-Through:

Interactive path definition by clicking
Smooth interpolated camera movement
Adjustable speed control
Loop playback


- Clipping Planes:

Three anatomical planes (Sagittal, Coronal, Axial)
Real-time cross-sectional views
Visual plane indicators (red, green, blue)
Independent plane control with sliders



### Advanced Features

- Smart Caching System:

NumPy-based compressed caching
10x faster subsequent loads
Automatic cache invalidation
~100MB cache for typical datasets


- Anatomical Color Schemes:

System-specific color palettes
Automatic structure identification
Customizable colors per part
Professional medical visualization standards


- Interactive Controls:

Individual part opacity control (0-100%)
Color picker for each structure
Focus mode (isolate specific parts)
Reset camera view


- Cinematic Lighting:

Multi-light setup for depth perception
Phong shading with specular highlights
Professional medical visualization quality



### MRI Viewer (mri_viewer.py)
- Core Views

Axial View (X-Y plane): Top-down brain slices
Coronal View (X-Z plane): Front-to-back slices
Sagittal View (Y-Z plane): Side view slices
Curved MPR View: Custom curved reconstructions

- MPR Features

Interactive Crosshairs: Synchronized across all views
Curved Planar MPR:

Click-to-define curved path on axial view
Z-slicing along the curve
Real-time interpolation



