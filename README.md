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

### • **Multi-System Support:**

![interface](https://github.com/rahmashraf/3D_Medical_Visualization/blob/main/assets/main.png)
___
### • **Surface Rendering:**

<p align="center">
  <img src="assets/Musculoskeletal.jpeg" width="250" />
  <img src="assets/heart.jpeg" width="250" />
  <br>
  <img src="assets/brain.jpeg" width="250" />
  <img src="assets/teeth.jpeg" width="250" />
</p>

___
### • **MRI Viewer (Curved MPR)**
 
**Curved MPR View**: Custom curved reconstructions

Click-to-define curved path on axial view
Z-slicing along the curve
Real-time interpolation
Display Options:
Full-height curved reconstruction
Dental arch visualization

![CurvedMpr](https://github.com/rahmashraf/3D_Medical_Visualization/blob/main/assets/curvedmpr.gif)

---
### • **Focus navigation:**

![FocusNavigation](https://github.com/rahmashraf/3D_Medical_Visualization/blob/main/assets/focusNavigation.gif)

### • **Heart Animation:**

Real-time cardiac cycle visualization
STL frame sequence support
30 FPS playback with parallel loading

### • **Clipping Planes:

Three anatomical planes (Sagittal, Coronal, Axial)
Real-time cross-sectional views
Visual plane indicators (red, green, blue)
Independent plane control with sliders


### • **Camera Fly-Through:

Interactive path definition by clicking
Smooth interpolated camera movement
Adjustable speed control
Loop playback

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



