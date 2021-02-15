import functools
import numpy as np
import k3d
from k3d.objects import Group, Line, Points
from k3d.plot import Plot
from .jointset import JointSet, MuPoTSJoints, OpenPoseJoints, CocoExJoints, PanopticJoints, Common14Joints
from typing import Union, NewType, Optional

SkeletonPlot = NewType('SkeletonPlot', Optional[Plot])
SkeletonObject = NewType('SkeletonObject', Optional[Group])
Color = NewType('Color', Union[str, int])


class Skeleton:
    def __init__(self, joint_coordinates: np.ndarray,
                 joint_set: JointSet, part_size: float, color: Color):
        self.joint_coordinates: np.ndarray = joint_coordinates
        self.joint_set: JointSet = joint_set
        self.part_size: float = part_size
        self.color: Color = color

    def to_skeleton_object(self) -> SkeletonObject:
        joint_points = self.get_joint_points()
        joint_lines = self.get_joint_lines()
        skeleton_object: SkeletonObject = SkeletonObject(Group())
        skeleton_object += joint_points
        return functools.reduce(
            lambda _skeleton_object, _skeleton_part: _skeleton_object + _skeleton_part,
            joint_lines,
            skeleton_object
        )

    def get_joint_colors(self) -> np.ndarray:
        joint_colors_shape = (self.joint_set.number_of_joints, )
        if self.color == 'default':
            joint_colors = np.zeros(shape=joint_colors_shape, dtype='uint32')
            joint_colors[self.joint_set.left_joint_indices] = 0xFF0000
            joint_colors[self.joint_set.right_joint_indices] = 0x0000FF
            joint_colors[self.joint_set.center_joint_indices] = 0xFFFFFF
            return joint_colors
        else:
            return np.full(shape=joint_colors_shape,
                           fill_value=self.color, dtype='uint32')

    def get_line_colors(self) -> np.ndarray:
        line_colors_shape = (self.joint_set.number_of_joints - 1, )
        if self.color == 'default':
            line_colors = np.zeros(shape=line_colors_shape, dtype='uint32')
            line_colors[self.joint_set.left_line_indices] = 0xFF0000
            line_colors[self.joint_set.right_line_indices] = 0x0000FF
            line_colors[self.joint_set.center_line_indices] = 0xFFFFFF
            return line_colors
        else:
            return np.full(shape=line_colors_shape,
                           fill_value=self.color, dtype='uint32')

    def get_joint_points(self) -> Points:
        skeleton_joint_colors = self.get_joint_colors()
        return k3d.points(
            positions=self.joint_coordinates, point_size=self.part_size,
            shader='mesh', colors=skeleton_joint_colors
        )

    def get_joint_lines(self) -> list[Line]:
        skeleton_line_colors = self.get_line_colors()
        return [self.__create_line_between_joints(
            start=self.joint_coordinates[self.joint_set.limb_graph[i][0]],
            end=self.joint_coordinates[self.joint_set.limb_graph[i][1]],
            width=self.part_size / 4.0,
            color=int(skeleton_line_colors[i])
        ) for i in range(len(self.joint_set.limb_graph))]

    @staticmethod
    def __create_line_between_joints(start: float, end: float, width: float, color: int) -> Line:
        return k3d.line(
            vertices=[start, end], shader='mesh',
            width=width, color=color
        )


class SkeletonVisualizer:
    def __init__(self, joint_set: JointSet, size_scalar: float = 1.0):
        self.joint_set: JointSet = joint_set
        self.plot: SkeletonPlot = None
        self.size_scalar: float = size_scalar

    def visualize(self, skeletons: np.ndarray, colors: list[Color] = None):
        self.__assert_visualization_arguments(skeletons, colors)
        number_of_skeletons = skeletons.shape[0]
        if colors is None:
            colors = ['default' for i in range(number_of_skeletons)]
        self.plot = self.__create_skeleton_plot(skeletons, colors)
        self.plot.display()

    def __assert_visualization_arguments(self, skeletons: np.ndarray, colors: list[Color]):
        assert len(skeletons.shape) == 3, 'The \'skeletons\' parameter should be a 3 dimensional numpy array.'
        if colors is not None:
            assert skeletons.shape[0] == len(colors)
        assert skeletons.shape[1] == self.joint_set.number_of_joints
        assert skeletons.shape[2] == 3

    def __create_skeleton_plot(self, skeletons: np.ndarray, colors: list[Color]) -> SkeletonPlot:
        skeleton_part_size = self.__calculate_skeleton_part_size(skeletons)
        skeleton_plot = k3d.plot()
        for skeleton in map(
                lambda skeleton_color_tuple: Skeleton(
                    joint_coordinates=skeleton_color_tuple[0],
                    joint_set=self.joint_set,
                    part_size=skeleton_part_size,
                    color=skeleton_color_tuple[1]),
                zip(skeletons, colors)):
            skeleton_plot += skeleton.to_skeleton_object()
        return skeleton_plot

    def __calculate_skeleton_part_size(self, skeletons: np.ndarray) -> float:
        max_values = [abs(skeleton).max() for skeleton in skeletons]
        return (min(max_values) / 50.0) * self.size_scalar


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