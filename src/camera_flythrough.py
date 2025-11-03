"""
Camera Fly-Through Module
=========================
Reusable classes for implementing cinematic camera fly-through in VTK applications.

Classes:
- CameraAnimator: Handles smooth camera path animation with cubic spline interpolation
- CustomInteractorStyle: VTK interactor that allows point picking for path definition
"""

import numpy as np
import vtk
from scipy.interpolate import CubicSpline


class CameraAnimator:
    """Handles smooth camera animations with easing"""

    def __init__(self, camera, easing_type='ease_in_out'):
        """
        Initialize camera animator

        Args:
            camera: vtkCamera object to animate
            easing_type: Type of easing ('ease_in_out', 'ease_in', 'ease_out', 'linear')
        """
        self.camera = camera
        self.path_points = []
        self.current_index = 0
        self.total_steps = 0
        self.easing_type = easing_type

    def set_path(self, points, steps_per_segment=20):
        """
        Set camera path with smooth cubic spline interpolation

        Args:
            points: List of (x, y, z) tuples representing waypoints
            steps_per_segment: Number of interpolated points between each waypoint
        """
        if len(points) < 2:
            self.path_points = points if points else []
            self.total_steps = len(self.path_points)
            self.current_index = 0
            return

        # Create smooth spline interpolation
        t = np.linspace(0, len(points) - 1, len(points))
        t_new = np.linspace(0, len(points) - 1, (len(points) - 1) * steps_per_segment + 1)

        cs_x = CubicSpline(t, [p[0] for p in points])
        cs_y = CubicSpline(t, [p[1] for p in points])
        cs_z = CubicSpline(t, [p[2] for p in points])

        self.path_points = np.column_stack([cs_x(t_new), cs_y(t_new), cs_z(t_new)])
        self.total_steps = len(self.path_points)
        self.current_index = 0

    def ease(self, t):
        """
        Apply easing function to parameter t

        Args:
            t: Parameter value between 0 and 1

        Returns:
            Eased value between 0 and 1
        """
        if self.easing_type == 'ease_in_out':
            return t * t * (3.0 - 2.0 * t)
        elif self.easing_type == 'ease_in':
            return t * t
        elif self.easing_type == 'ease_out':
            return t * (2.0 - t)
        return t  # linear

    def step(self, loop=True):
        """
        Advance camera one step along path

        Args:
            loop: If True, loop back to start when path ends

        Returns:
            True if camera moved, False if path ended (and not looping)
        """
        if len(self.path_points) == 0:
            return False

        if self.current_index >= len(self.path_points):
            if loop:
                self.current_index = 0
            else:
                return False

        # Current position
        pos = self.path_points[self.current_index]

        # Look ahead for focal point (smoother motion)
        look_ahead = min(5, len(self.path_points) - self.current_index - 1)
        if look_ahead > 0:
            focal = self.path_points[self.current_index + look_ahead]
        else:
            focal = self.path_points[-1]
            if np.array_equal(pos, focal) and len(self.path_points) > 1:
                focal = self.path_points[-2]

        # Update camera
        self.camera.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
        self.camera.SetFocalPoint(float(focal[0]), float(focal[1]), float(focal[2]))

        self.current_index += 1
        return True

    def reset(self):
        """Reset animation to start of path"""
        self.current_index = 0

    def get_progress(self):
        """
        Get current animation progress

        Returns:
            Float between 0 and 1 representing progress
        """
        if self.total_steps == 0:
            return 0.0
        return self.current_index / self.total_steps


class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    Custom interactor style to handle point picking for path definition.
    Inherits from TrackballCamera to keep zoom (right-mouse) and pan (middle-mouse).
    """

    def __init__(self, parent=None):
        """
        Initialize custom interactor

        Args:
            parent: Parent widget that has path_definition_mode attribute and picker
        """
        self.parent = parent
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.on_left_button_down, 1.0)

    def on_left_button_down(self, obj, event):
        """
        Handle left mouse button click

        If in path definition mode, picks 3D point on model.
        Otherwise, performs normal camera rotation.
        """
        # If in path definition mode, hijack the click for picking
        if self.parent and self.parent.path_definition_mode:
            click_pos = self.GetInteractor().GetEventPosition()

            # Use the parent's picker
            self.parent.picker.Pick(click_pos[0], click_pos[1], 0, self.parent.renderer)

            # Check if we hit an actor
            if self.parent.picker.GetCellId() >= 0:
                pick_coords = self.parent.picker.GetPickPosition()
                self.parent.fly_path_points.append(pick_coords)

                # Update the visual guide (the path)
                if hasattr(self.parent, 'update_path_visual'):
                    self.parent.update_path_visual()

                # Render the scene
                self.GetInteractor().Render()
        else:
            # Not in path mode, so just do the normal camera rotation
            super().OnLeftButtonDown()


class PathVisualizer:
    """Helper class to visualize the fly-through path with spheres and lines"""

    @staticmethod
    def create_path_actor(path_points, renderer, sphere_radius=None):
        """
        Create a VTK actor showing the path as red spheres connected by yellow lines

        Args:
            path_points: List of (x, y, z) tuples
            renderer: vtkRenderer to compute appropriate sphere size
            sphere_radius: Optional fixed sphere radius, auto-calculated if None

        Returns:
            vtkAssembly containing sphere and line actors
        """
        if not path_points:
            return None

        # Create points
        points_data = vtk.vtkPolyData()
        points = vtk.vtkPoints()

        for p in path_points:
            points.InsertNextPoint(p)
        points_data.SetPoints(points)

        # Create spheres at each point
        sphere_source = vtk.vtkSphereSource()

        # Calculate appropriate radius
        if sphere_radius is None:
            bounds = renderer.ComputeVisiblePropBounds()
            diag = np.sqrt(
                (bounds[1] - bounds[0]) ** 2 +
                (bounds[3] - bounds[2]) ** 2 +
                (bounds[5] - bounds[4]) ** 2
            )
            sphere_radius = diag * 0.01 if diag > 0 else 1.0

        sphere_source.SetRadius(sphere_radius)

        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(sphere_source.GetOutputPort())
        glyph.SetInputData(points_data)
        glyph.Update()

        points_mapper = vtk.vtkPolyDataMapper()
        points_mapper.SetInputConnection(glyph.GetOutputPort())

        points_actor = vtk.vtkActor()
        points_actor.SetMapper(points_mapper)
        points_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red

        # Create lines connecting points
        lines_actor = vtk.vtkActor()
        if len(path_points) > 1:
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(len(path_points))
            for i in range(len(path_points)):
                lines.InsertCellPoint(i)

            lines_data = vtk.vtkPolyData()
            lines_data.SetPoints(points)
            lines_data.SetLines(lines)

            lines_mapper = vtk.vtkPolyDataMapper()
            lines_mapper.SetInputData(lines_data)

            lines_actor.SetMapper(lines_mapper)
            lines_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow
            lines_actor.GetProperty().SetLineWidth(3)

        # Combine into assembly
        assembly = vtk.vtkAssembly()
        assembly.AddPart(points_actor)
        assembly.AddPart(lines_actor)

        return assembly


class ClippingPlaneManager:
    """Manages a clipping plane that follows the camera during fly-through"""

    def __init__(self, renderer):
        """
        Initialize clipping plane manager

        Args:
            renderer: vtkRenderer to get camera from
        """
        self.renderer = renderer
        self.clip_plane = None
        self.active = False
        self.tracked_actors = []

    def enable(self, actors):
        """
        Enable clipping plane for given actors

        Args:
            actors: List of vtkActor objects to apply clipping to
        """
        if not self.clip_plane:
            self.clip_plane = vtk.vtkPlane()

        self.tracked_actors = actors

        for actor in actors:
            mapper = actor.GetMapper()
            if mapper:
                mapper.AddClippingPlane(self.clip_plane)

        self.active = True
        self.update()

    def disable(self):
        """Disable and remove clipping plane from all actors"""
        if self.clip_plane:
            for actor in self.tracked_actors:
                mapper = actor.GetMapper()
                if mapper:
                    mapper.RemoveClippingPlane(self.clip_plane)

        self.active = False
        self.tracked_actors = []

    def update(self, offset=0.1):
        """
        Update clipping plane to match camera position and orientation

        Args:
            offset: Distance offset in front of camera (to avoid clipping camera itself)
        """
        if not self.active or not self.clip_plane:
            return

        camera = self.renderer.GetActiveCamera()
        cam_normal = camera.GetViewPlaneNormal()
        cam_pos = camera.GetPosition()

        # Offset the plane slightly in front of the camera
        offset_pos = (
            cam_pos[0] + cam_normal[0] * offset,
            cam_pos[1] + cam_normal[1] * offset,
            cam_pos[2] + cam_normal[2] * offset
        )

        self.clip_plane.SetOrigin(offset_pos)
        self.clip_plane.SetNormal(cam_normal)