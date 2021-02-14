import numpy as np
import k3d
from k3d.objects import Group, Line, Points
from k3d.plot import Plot
from .jointset import JointSet, MuPoTSJoints, OpenPoseJoints, CocoExJoints, PanopticJoints, Common14Joints
from typing import Union, NewType

SkeletonPlot = NewType('SkeletonPlot', Union[Plot, None])
Color = NewType('Color', Union[str, int])
Colors = NewType('Colors', Union[list[Color], None])


class Skeleton:
    def __init__(self, joint_coordinates: np.ndarray, part_size: float, color: Color):
        self.joint_coordinates: np.ndarray = joint_coordinates
        self.part_size: float = part_size
        self.color: Color = color


class SkeletonVisualizer:
    def __init__(self, joint_set: JointSet, size_scalar: float = 1.0):
        self.joint_set: JointSet = joint_set
        self.plot: SkeletonPlot = None
        self.size_scalar: float = size_scalar

    def visualize(self, skeletons: np.ndarray, colors: Colors = None):
        self.__assert_visualization_arguments(skeletons, colors)
        number_of_skeletons = skeletons.shape[0]
        if colors is None:
            colors = ['default' for i in range(number_of_skeletons)]
        skeleton_part_size = self.__calculate_size_from_skeletons(skeletons)
        self.plot = k3d.plot()
        for i in range(number_of_skeletons):
            skeleton = Skeleton(
                joint_coordinates=skeletons[i],
                part_size=skeleton_part_size,
                color=colors[i])
            self.__add_skeleton_to_plot(skeleton)
        self.plot.display()

    def __assert_visualization_arguments(self, skeletons: np.ndarray, colors: Colors):
        assert len(skeletons.shape) == 3, 'The \'skeletons\' parameter should be a 3 dimensional numpy array.'
        if colors is not None:
            assert skeletons.shape[0] == len(colors)
        assert skeletons.shape[1] == self.joint_set.number_of_joints
        assert skeletons.shape[2] == 3

    def __calculate_size_from_skeletons(self, skeletons: np.ndarray) -> float:
        max_values = [abs(skeleton).max() for skeleton in skeletons]
        return (min(max_values) / 50.0) * self.size_scalar

    def __add_skeleton_to_plot(self, skeleton: Skeleton):
        skeleton_object = self.__generate_skeleton_object(skeleton)
        self.plot += skeleton_object

    def __generate_skeleton_object(self, skeleton: Skeleton) -> Group:
        assert skeleton.joint_coordinates.shape == (self.joint_set.number_of_joints, 3)
        skeleton_joint_colors, skeleton_line_colors = self.__get_skeleton_colors(skeleton)
        skeleton_joint_points = self.__create_skeleton_joint_points(skeleton, skeleton_joint_colors)
        skeleton_joint_lines = self.__create_skeleton_lines(skeleton, skeleton_line_colors)
        skeleton_object: Group = Group()
        skeleton_object += skeleton_joint_points
        for line in skeleton_joint_lines:
            skeleton_object += line
        return skeleton_object

    def __get_skeleton_colors(self, skeleton: Skeleton) -> tuple[np.ndarray, np.ndarray]:
        joint_colors_shape = (self.joint_set.number_of_joints, )
        line_colors_shape = (self.joint_set.number_of_joints - 1, )
        if skeleton.color == 'default':
            joint_colors = np.zeros(shape=joint_colors_shape, dtype='uint32')
            line_colors = np.zeros(shape=line_colors_shape, dtype='uint32')
            joint_colors[self.joint_set.left_joint_indices] = 0xFF0000
            joint_colors[self.joint_set.right_joint_indices] = 0x0000FF
            joint_colors[self.joint_set.center_joint_indices] = 0xFFFFFF
            line_colors[self.joint_set.left_line_indices] = 0xFF0000
            line_colors[self.joint_set.right_line_indices] = 0x0000FF
            line_colors[self.joint_set.center_line_indices] = 0xFFFFFF
            return joint_colors, line_colors
        else:
            return np.full(shape=joint_colors_shape, fill_value=skeleton.color, dtype='uint32'),\
                   np.full(shape=line_colors_shape, fill_value=skeleton.color, dtype='uint32')

    @staticmethod
    def __create_skeleton_joint_points(skeleton: Skeleton, skeleton_joint_colors: np.ndarray) -> Points:
        return k3d.points(
            positions=skeleton.joint_coordinates, point_size=skeleton.part_size,
            shader='mesh', colors=skeleton_joint_colors
        )

    def __create_skeleton_lines(self, skeleton: Skeleton, skeleton_line_colors: np.ndarray):
        return [self.__create_line_between_joints(
            start=skeleton.joint_coordinates[self.joint_set.limb_graph[i][0]],
            end=skeleton.joint_coordinates[self.joint_set.limb_graph[i][1]],
            width=skeleton.part_size / 4.0,
            color=int(skeleton_line_colors[i])
        ) for i in range(len(self.joint_set.limb_graph))]

    @staticmethod
    def __create_line_between_joints(start: float, end: float, width: float, color: int) -> Line:
        return k3d.line(
            vertices=[start, end], shader='mesh',
            width=width, color=color
        )


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