import pickle
from typing import Optional, List, Dict, Callable, IO

import ipywidgets as widgets
import k3d
import numpy as np
from IPython.display import display
from ipywidgets import Play, Checkbox, IntSlider, Button
from k3d.plot import Plot

from .jointset import JointSet, MuPoTSJoints, OpenPoseJoints, CocoExJoints, PanopticJoints, Common14Joints
from .objects import Skeleton, DrawableSkeleton, Color, Positions


class FrameUpdater:
    def __init__(self, video_player: Play):
        self.video_player: Play = video_player

    def update_to_next_frame(self, button) -> None:
        if self.video_player.max > self.video_player.value:
            self.video_player.value += 1

    def update_to_previous_frame(self, button) -> None:
        if self.video_player.min < self.video_player.value:
            self.video_player.value -= 1


class SkeletonVisualizer:
    def __init__(self, joint_set: JointSet, size_scalar: float = 1.0):
        self.joint_set: JointSet = joint_set
        self.plot: Optional[Plot] = None
        self.size_scalar: float = size_scalar
        self.skeletons: List[DrawableSkeleton] = []
        self.joint_names_visible: Checkbox = Checkbox(description="Show Joint Names")
        self.joint_coordinates_visible: Checkbox = Checkbox(description="Show Coordinates")

    def visualize(self, skeletons: np.ndarray, colors: List[Color] = None,
                  include_names: bool = False, include_coordinates: bool = False) -> None:
        self.__assert_skeleton_shapes(skeletons)
        colors = self.__init_colors(skeletons.shape[0], colors)
        self.__assert_include_arguments(include_names, include_coordinates)
        if include_names:
            skeleton_converter = Skeleton.to_drawable_skeleton_with_names
        elif include_coordinates:
            skeleton_converter = Skeleton.to_drawable_skeleton_with_coordinates
        else:
            skeleton_converter = Skeleton.to_drawable_skeleton
        self.__create_skeleton_plot(skeletons, skeleton_converter, colors)
        self.plot.display()
        self.__display_checkboxes(include_names, include_coordinates)

    def visualize_video_from_file(self, file_name: str, colors: List[Color] = None, fps: int = 15,
                                  include_names: bool = False, include_coordinates: bool = False) -> None:
        file: IO = open(file_name, 'rb')
        frames = pickle.load(file)
        file.close()
        self.visualize_video(frames, colors, fps, include_names, include_coordinates)

    def visualize_video(self, frames: np.ndarray, colors: List[Color] = None, fps: int = 15,
                        include_names: bool = False, include_coordinates: bool = False) -> None:
        assert len(frames.shape) == 4
        first_frame: np.ndarray = frames[0]
        self.__assert_skeleton_shapes(first_frame)
        colors = self.__init_colors(first_frame.shape[0], colors)
        skeleton_timestamps: List[Dict[str, np.ndarray]] = self.__get_skeleton_positions_timestamps(frames)
        self.__assert_include_arguments(include_names, include_coordinates)
        if include_names:
            skeleton_converter = Skeleton.to_drawable_skeleton_for_video_with_names
        elif include_coordinates:
            skeleton_converter = Skeleton.to_drawable_skeleton_for_video_with_coordinates
        else:
            skeleton_converter = Skeleton.to_drawable_skeleton_for_video
        self.__create_skeleton_plot(first_frame, skeleton_converter, colors, skeleton_timestamps)
        self.plot.display()
        self.__display_video_player(fps, frames)
        self.__display_checkboxes(include_names, include_coordinates)

    @staticmethod
    def __get_skeleton_positions_timestamps(frames: np.ndarray) -> List[Dict[str, np.ndarray]]:
        frames_swapped: np.ndarray = np.swapaxes(frames, 0, 1)
        return [{
            str(timestamp_index): frames_swapped[current_frame_index][timestamp_index]
            for timestamp_index in range(len(frames_swapped[current_frame_index]))
        } for current_frame_index in range(len(frames_swapped))]

    def __create_skeleton_plot(self, skeletons: np.ndarray, skeleton_converter: Callable[[Skeleton], DrawableSkeleton],
                               colors: List[Color], positions: Optional[Positions] = None) -> None:
        self.__init_skeleton_plot(skeletons)
        skeleton_part_size: float = self.__calculate_skeleton_part_size(skeletons)
        if positions is None:
            positions = skeletons
        skeletons: List[Skeleton] = self.__get_skeletons(positions, colors, skeleton_part_size)
        drawable_skeletons: List[DrawableSkeleton] = [skeleton_converter(skeleton) for skeleton in skeletons]
        self.__add_skeletons_to_plot(drawable_skeletons)

    def __init_skeleton_plot(self, skeletons: np.ndarray) -> None:
        self.plot = k3d.plot(antialias=1, camera_auto_fit=False)
        centroid: np.ndarray = np.average(np.average(skeletons, axis=0), axis=0)
        camera_up_vector: np.ndarray = np.zeros(shape=(3,))
        for line_indices in self.joint_set.vertically_aligned_line_indices:
            camera_up_vector += np.sum(
                skeletons[:, line_indices[1]] - skeletons[:, line_indices[0]],
                axis=0)
        camera_up_vector /= np.linalg.norm(camera_up_vector, ord=2)
        self.plot.camera = [
            0, 0, 0,
            centroid[0], centroid[1], centroid[2],
            camera_up_vector[0], camera_up_vector[1], camera_up_vector[2]]

    def __calculate_skeleton_part_size(self, skeletons: np.ndarray) -> float:
        max_values = [abs(skeleton).max() for skeleton in skeletons]
        return (min(max_values) / 100.0) * self.size_scalar

    def __get_skeletons(self, positions: List[Positions], colors: List[Color],
                        skeleton_part_size: float) -> List[Skeleton]:
        return list(map(lambda position_color_tuple: Skeleton(
            joint_positions=position_color_tuple[0],
            joint_set=self.joint_set,
            part_size=skeleton_part_size,
            color=position_color_tuple[1]
        ), zip(positions, colors)))

    def __add_skeletons_to_plot(self, skeletons: List[DrawableSkeleton]) -> None:
        for drawable_skeleton in skeletons:
            self.plot += drawable_skeleton
            self.skeletons.append(drawable_skeleton)

    def __display_video_player(self, fps, frames) -> None:
        video_player: Play = Play(
            value=0,
            min=0, max=frames.shape[0] - 1,
            step=1, interval=1000 / fps,
            description='Press play', disabled=False)
        frame_updater = FrameUpdater(video_player)
        frame_slider = IntSlider(value=0, min=0, max=frames.shape[0] - 1, step=1, description='Frame')
        next_frame_button = Button(description='Next frame')
        next_frame_button.on_click(frame_updater.update_to_next_frame)
        previous_frame_button = Button(description='Previous frame')
        previous_frame_button.on_click(frame_updater.update_to_previous_frame)
        widgets.jslink((video_player, 'value'), (frame_slider, 'value'))
        widgets.jslink((video_player, 'value'), (self.plot, 'time'))
        display(widgets.HBox([video_player, previous_frame_button, frame_slider, next_frame_button]))

    @staticmethod
    def __init_colors(number_of_skeletons: int, colors: List[Color]) -> List[Color]:
        if colors is None:
            colors = ['default' for _ in range(number_of_skeletons)]
        else:
            assert number_of_skeletons == len(colors), \
                'The \'skeletons\' and \'colors\' parameters must be the same length.'
        return colors

    def __assert_skeleton_shapes(self, skeletons: np.ndarray) -> None:
        assert len(skeletons.shape) == 3, 'The \'skeletons\' parameter should be a 3 dimensional numpy array.'
        assert skeletons.shape[1] == self.joint_set.number_of_joints, \
            'The number of joints of skeletons and the number of joints in the specified joint set must be the same.'
        assert skeletons.shape[2] == 3, 'The skeleton joint coordinates must be 3 dimensional'

    @staticmethod
    def __assert_include_arguments(include_names: bool, include_coordinates) -> None:
        if include_names is True and include_coordinates is True:
            raise AttributeError('Either names or coordinates can be showed, but not both.')

    def __display_checkboxes(self, include_names: bool, include_coordinates: bool) -> None:
        if include_names:
            self.__link_joint_name_visibility_with_checkbox()
            display(self.joint_names_visible)
        if include_coordinates:
            self.__link_joint_coordinate_visibility_with_checkbox()
            display(self.joint_coordinates_visible)

    def __link_joint_name_visibility_with_checkbox(self) -> None:
        for skeleton in self.skeletons:
            for joint_name in skeleton.joint_names:
                widgets.jslink((joint_name, 'visible'), (self.joint_names_visible, 'value'))

    def __link_joint_coordinate_visibility_with_checkbox(self) -> None:
        for skeleton in self.skeletons:
            for joint_coordinate in skeleton.joint_coordinates:
                widgets.jslink((joint_coordinate, 'visible'), (self.joint_coordinates_visible, 'value'))


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
