from typing import Optional, List

import ipywidgets as widgets
import k3d
import numpy as np
from IPython.display import display
from k3d.plot import Plot

from .jointset import JointSet, MuPoTSJoints, OpenPoseJoints, CocoExJoints, PanopticJoints, Common14Joints
from .objects import Skeleton, DrawableSkeleton, Color


class SkeletonVisualizer:
    def __init__(self, joint_set: JointSet, size_scalar: float = 1.0):
        self.joint_set: JointSet = joint_set
        self.plot: Optional[Plot] = None
        self.size_scalar: float = size_scalar
        self.skeletons: List[DrawableSkeleton] = []
        self.joint_names_visible = widgets.Checkbox(description="Show Joint Names")

    def visualize_video(self, frames: np.ndarray, colors: List[Color] = None):
        assert len(frames.shape) == 4
        first_frame = frames[0]
        self.__assert_skeleton_shapes(first_frame)
        colors = self.__init_colors(frames.shape[1], colors)
        self.plot = self.__create_skeleton_plot(first_frame, colors)
        self.plot.display()

    def visualize(self, skeletons: np.ndarray, colors: List[Color] = None):
        self.__assert_skeleton_shapes(skeletons)
        colors = self.__init_colors(skeletons.shape[0], colors)
        self.plot = self.__create_skeleton_plot(skeletons, colors)
        self.plot.display()

    def visualize_with_names(self, skeletons: np.ndarray, colors: List[Color] = None):
        self.__assert_skeleton_shapes(skeletons)
        colors = self.__init_colors(skeletons.shape[0], colors)
        self.plot = self.__create_skeleton_plot(skeletons, colors, include_names=True)
        self.__link_joint_name_visibility_with_checkbox()
        self.plot.display()
        display(self.joint_names_visible)

    @staticmethod
    def __init_colors(number_of_skeletons: int, colors: List[Color]):
        if colors is None:
            colors = ['default' for _ in range(number_of_skeletons)]
        else:
            assert number_of_skeletons == len(colors), \
                'The \'skeletons\' and \'colors\' parameters must be the same length.'
        return colors

    def __assert_skeleton_shapes(self, skeletons: np.ndarray):
        assert len(skeletons.shape) == 3, 'The \'skeletons\' parameter should be a 3 dimensional numpy array.'
        assert skeletons.shape[1] == self.joint_set.number_of_joints,\
            'The number of joints of skeletons and the number of joints in the specified joint set must be the same.'
        assert skeletons.shape[2] == 3, 'The skeleton joint coordinates must be 3 dimensional'

    def __create_skeleton_plot(self, skeletons: np.ndarray,
                               colors: List[Color], include_names: bool = False) -> Optional[Plot]:
        skeleton_part_size = self.__calculate_skeleton_part_size(skeletons)
        skeleton_plot = k3d.plot()
        for skeleton in map(
                lambda skeleton_color_tuple: Skeleton(
                    joint_coordinates=skeleton_color_tuple[0],
                    joint_set=self.joint_set,
                    part_size=skeleton_part_size,
                    color=skeleton_color_tuple[1]),
                zip(skeletons, colors)):
            if include_names:
                drawable_skeleton = skeleton.to_drawable_skeleton_with_names()
            else:
                drawable_skeleton = skeleton.to_drawable_skeleton()
            skeleton_plot += drawable_skeleton
            self.skeletons.append(drawable_skeleton)
        return skeleton_plot

    def __calculate_skeleton_part_size(self, skeletons: np.ndarray) -> float:
        max_values = [abs(skeleton).max() for skeleton in skeletons]
        return (min(max_values) / 100.0) * self.size_scalar

    def __link_joint_name_visibility_with_checkbox(self):
        for skeleton in self.skeletons:
            for joint_name in skeleton.joint_names:
                widgets.jslink((joint_name, 'visible'), (self.joint_names_visible, 'value'))


class MuPoTSVisualizer(SkeletonVisualizer):
    def __init__(self, size_scalar: float = 1.0):
        super().__init__(joint_set=MuPoTSJoints(), size_scalar=size_scalar)


class OpenPoseVisualizer(SkeletonVisualizer):
    def __init__(self, size_scalar: float = 1.0):
        super().__init__(joint_set=OpenPoseJoints(), size_scalar=size_scalar)


class CocoExVisualizer(SkeletonVisualizer):
    def __init__(self, size_scalar: float = 1.0):
        super().__init__(joint_set=CocoExJoints(), size_scalar=size_scalar)


class PanopticVisualizer(SkeletonVisualizer):
    def __init__(self, size_scalar: float = 1.0):
        super().__init__(joint_set=PanopticJoints(), size_scalar=size_scalar)


class Common14Visualizer(SkeletonVisualizer):
    def __init__(self, size_scalar: float = 1.0):
        super().__init__(joint_set=Common14Joints(), size_scalar=size_scalar)
