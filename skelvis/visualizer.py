import pickle
from typing import Optional, List

import ipywidgets as widgets
import k3d
import numpy as np
from IPython.display import display
from ipywidgets import Play
from k3d.plot import Plot

from .jointset import JointSet, MuPoTSJoints, OpenPoseJoints, CocoExJoints, PanopticJoints, Common14Joints
from .objects import Skeleton, DrawableSkeleton, Color


class FrameUpdater:
    def __init__(self, skeletons: List[DrawableSkeleton], frames: np.ndarray, joint_set: JointSet, video_player: Play):
        self.skeletons = skeletons
        self.frames = frames
        self.joint_set: JointSet = joint_set
        self.video_player: Play = video_player

    def update_plot(self, frame_index) -> None:
        current_frame = self.frames[frame_index.new]
        number_of_skeletons: int = len(self.skeletons)
        for i in range(number_of_skeletons):
            for j in range(len(self.joint_set.limb_graph)):
                line_indices = self.joint_set.limb_graph[j]
                self.skeletons[i].joint_lines[j].vertices = \
                    (current_frame[i][line_indices[0]], current_frame[i][line_indices[1]])
            self.skeletons[i].joint_points.positions = current_frame[i]

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

    def visualize_video_from_file(self, file_name: str, colors: List[Color] = None, fps: int = 15) -> None:
        file = open(file_name, 'rb')
        frames = pickle.load(file)
        file.close()
        self.visualize_video(frames, colors, fps)

    def visualize_video(self, frames: np.ndarray, colors: List[Color] = None, fps: int = 15) -> None:
        assert len(frames.shape) == 4
        first_frame = frames[0]
        self.__assert_skeleton_shapes(first_frame)
        colors = self.__init_colors(frames.shape[1], colors)
        self.plot = self.__create_skeleton_plot(first_frame, colors)
        self.plot.display()
        video_player: Play = Play(
            value=0,
            min=0, max=frames.shape[0] - 1,
            step=1, interval = 1000 / fps,
            description='Press play', disabled=False
        )
        frame_updater = FrameUpdater(self.skeletons, frames, self.joint_set, video_player)
        video_player.observe(frame_updater.update_plot, names='value')
        frame_slider = widgets.IntSlider(value=0, min=0, max=frames.shape[0], step=1, description='Frame')
        next_frame_button = widgets.Button(description='Next frame')
        next_frame_button.on_click(frame_updater.update_to_next_frame)
        previous_frame_button = widgets.Button(description='Prev frame')
        previous_frame_button.on_click(frame_updater.update_to_previous_frame)
        widgets.jslink((video_player, 'value'), (frame_slider, 'value'))
        display(widgets.HBox([video_player, previous_frame_button, frame_slider, next_frame_button]))


    def visualize(self, skeletons: np.ndarray, colors: List[Color] = None) -> None:
        self.__assert_skeleton_shapes(skeletons)
        colors = self.__init_colors(skeletons.shape[0], colors)
        self.plot = self.__create_skeleton_plot(skeletons, colors)
        self.plot.display()

    def visualize_with_names(self, skeletons: np.ndarray, colors: List[Color] = None) -> None:
        self.__assert_skeleton_shapes(skeletons)
        colors = self.__init_colors(skeletons.shape[0], colors)
        self.plot = self.__create_skeleton_plot(skeletons, colors, include_names=True)
        self.__link_joint_name_visibility_with_checkbox()
        self.plot.display()
        display(self.joint_names_visible)

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
