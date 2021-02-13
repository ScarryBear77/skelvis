import k3d


class Skeleton:
    def __init__(self, joint_coordinates, part_size, color):
        self.joint_coordinates = joint_coordinates
        self.part_size = part_size
        self.color = color


class SkeletonVisualizer:
    def __init__(self, joint_set, size_scalar=1.0):
        self.joint_set = joint_set
        self.plot = None
        self.size_scalar = size_scalar

    def visualize(self, skeletons, colors=None):
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

    def __calculate_size_from_skeletons(self, skeletons):
        max_values = [abs(skeleton).max() for skeleton in skeletons]
        return (min(max_values) / 50.0) * self.size_scalar

    def __assert_visualization_arguments(self, skeletons, colors):
        assert len(skeletons.shape) == 3, 'The \'skeletons\' parameter should be a 3 dimensional numpy array.'
        if colors is not None:
            assert skeletons.shape[0] == len(colors)
        assert skeletons.shape[1] == self.joint_set.number_of_joints
        assert skeletons.shape[2] == 3

    def __add_skeleton_to_plot(self, skeleton):
        skeleton_plot = self.joint_set.generate_skeleton_plot(skeleton)
        self.plot += skeleton_plot
