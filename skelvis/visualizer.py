import k3d
import numpy as np
from .jointset import JointSet, Skeleton, SkeletonPlot


class SkeletonVisualizer:
    def __init__(self, joint_set: JointSet, size_scalar: float = 1.0):
        self.joint_set: JointSet = joint_set
        self.plot: SkeletonPlot = None
        self.size_scalar: float = size_scalar

    def visualize(self, skeletons: np.ndarray, colors=None):
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
                color=colors[i]
            )
            self.__add_skeleton_to_plot(skeleton)
        self.plot.display()

    def __calculate_size_from_skeletons(self, skeletons: np.ndarray) -> float:
        max_values = [abs(skeleton).max() for skeleton in skeletons]
        return (min(max_values) / 50.0) * self.size_scalar

    def __assert_visualization_arguments(self, skeletons: np.ndarray, colors):
        assert len(skeletons.shape) == 3, 'The \'skeletons\' parameter should be a 3 dimensional numpy array.'
        if colors is not None:
            assert skeletons.shape[0] == len(colors)
        assert skeletons.shape[1] == self.joint_set.number_of_joints
        assert skeletons.shape[2] == 3

    def __add_skeleton_to_plot(self, skeleton: Skeleton):
        skeleton_plot = self.joint_set.generate_skeleton_plot(skeleton)
        self.plot += skeleton_plot
