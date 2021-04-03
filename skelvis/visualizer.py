import pickle
from typing import Optional, List, Dict, Callable

import ipywidgets as widgets
import k3d
import numpy as np
from IPython.display import display
from ipywidgets import Play
from k3d.plot import Plot

from .jointset import JointSet, MuPoTSJoints, OpenPoseJoints, CocoExJoints, PanopticJoints, Common14Joints
from .objects import Skeleton, DrawableSkeleton, Color, Positions


class FrameUpdater:
    def __init__(self, skeletons: List[DrawableSkeleton], frames: np.ndarray, joint_set: JointSet, video_player: Play):
        self.skeletons = skeletons
        self.frames = frames
        self.joint_set: JointSet = joint_set
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
        self.joint_names_visible = widgets.Checkbox(description="Show Joint Names")

    def visualize(self, skeletons: np.ndarray, colors: List[Color] = None, include_names: bool = False) -> None:
        self.__assert_skeleton_shapes(skeletons)
        colors = self.__init_colors(skeletons.shape[0], colors)
        if include_names:
            skeleton_converter = Skeleton.to_drawable_skeleton_with_names
        else:
            skeleton_converter = Skeleton.to_drawable_skeleton
        self.__create_skeleton_plot(skeletons, skeleton_converter, colors)
        self.plot.display()
        if include_names:
            self.__link_joint_name_visibility_with_checkbox()
            display(self.joint_names_visible)

    def visualize_video_from_file(
            self, file_name: str, colors: List[Color] = None,
            fps: int = 15, include_names: bool = False) -> None:
        file = open(file_name, 'rb')
        frames = pickle.load(file)
        file.close()
        self.visualize_video(frames, colors, fps, include_names)

    def visualize_video(
            self, frames: np.ndarray, colors: List[Color] = None,
            fps: int = 15, include_names: bool = False) -> None:
        assert len(frames.shape) == 4
        first_frame = frames[0]
        self.__assert_skeleton_shapes(first_frame)
        colors = self.__init_colors(first_frame.shape[0], colors)
        skeleton_timestamps: List[Dict[str, np.ndarray]] = self.__get_skeleton_positions_timestamps(frames)
        self.__create_skeleton_plot(
            skeletons=first_frame,
            skeleton_converter=Skeleton.to_drawable_skeleton_for_video,
            colors=colors, positions=skeleton_timestamps)
        self.plot.display()
        self.__display_video_player(fps, frames)

    @staticmethod
    def __get_skeleton_positions_timestamps(frames: np.ndarray) -> List[Dict[str, np.ndarray]]:
        frames_swapped = np.swapaxes(frames, 0, 1)
        return [
            {
                str(timestamp_index): frames_swapped[current_frame_index][timestamp_index]
                for timestamp_index in range(len(frames_swapped[current_frame_index]))
            }
            for current_frame_index in range(len(frames_swapped))
        ]

    def __create_skeleton_plot(self, skeletons: np.ndarray, skeleton_converter: Callable[[Skeleton], DrawableSkeleton],
                               colors: List[Color], positions: Optional[Positions] = None) -> None:
        self.__init_skeleton_plot(skeletons)
        skeleton_part_size = self.__calculate_skeleton_part_size(skeletons)
        if positions is None:
            positions = skeletons
        skeletons: List[Skeleton] = self.__get_skeletons(positions, colors, skeleton_part_size)
        drawable_skeletons: List[DrawableSkeleton] = [skeleton_converter(skeleton) for skeleton in skeletons]
        self.__add_skeletons_to_plot(drawable_skeletons)

    def __init_skeleton_plot(self, skeletons: np.ndarray):
        self.plot = k3d.plot(camera_auto_fit=False)
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

    def __get_skeletons(self, positions: List[Positions],
                        colors: List[Color], skeleton_part_size: float) -> List[Skeleton]:
        return list(map(lambda position_color_tuple: Skeleton(
            joint_positions=position_color_tuple[0],
            joint_set=self.joint_set,
            part_size=skeleton_part_size,
            color=position_color_tuple[1]
        ), zip(positions, colors)))

    def __add_skeletons_to_plot(self, skeletons: List[DrawableSkeleton]):
        for drawable_skeleton in skeletons:
            self.plot += drawable_skeleton
            self.skeletons.append(drawable_skeleton)

    def __display_video_player(self, fps, frames):
        video_player: Play = Play(
            value=0,
            min=0, max=frames.shape[0] - 1,
            step=1, interval=1000 / fps,
            description='Press play', disabled=False
        )
        frame_updater = FrameUpdater(self.skeletons, frames, self.joint_set, video_player)
        frame_slider = widgets.IntSlider(value=0, min=0, max=frames.shape[0] - 1, step=1, description='Frame')
        next_frame_button = widgets.Button(description='Next frame')
        next_frame_button.on_click(frame_updater.update_to_next_frame)
        previous_frame_button = widgets.Button(description='Prev frame')
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
        assert skeletons.shape[1] == self.joint_set.number_of_joints,\
            'The number of joints of skeletons and the number of joints in the specified joint set must be the same.'
        assert skeletons.shape[2] == 3, 'The skeleton joint coordinates must be 3 dimensional'

    def __link_joint_name_visibility_with_checkbox(self) -> None:
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
