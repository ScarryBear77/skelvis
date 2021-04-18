import pickle
from typing import Optional, List, Dict, Callable, IO, Tuple

import ipywidgets as widgets
import k3d
import numpy as np
from IPython.display import display
from ipywidgets import Play, Checkbox, IntSlider, Button, ColorPicker, VBox, Tab
from k3d.plot import Plot

from .jointset import JointSet, MuPoTSJoints, OpenPoseJoints, CocoExJoints, PanopticJoints, Common14Joints
from .objects import Skeleton, DrawableSkeleton, Color, Positions


class VideoPlayer:
    def __init__(self, plot: Plot):
        self.plot: Plot = plot
        self.video_player: Optional[Play] = None

    def display_video_player(self, fps: int, frames: np.ndarray) -> None:
        self.video_player = Play(
            value=0,
            min=0, max=frames.shape[0] - 1,
            step=1, interval=1000 / fps,
            description='Press play', disabled=False)
        frame_slider = IntSlider(value=0, min=0, max=frames.shape[0] - 1, step=1, description='Frame')
        next_frame_button = Button(description='Next frame')
        next_frame_button.on_click(self.__update_to_next_frame)
        previous_frame_button = Button(description='Previous frame')
        previous_frame_button.on_click(self.__update_to_previous_frame)
        widgets.jslink((self.video_player, 'value'), (frame_slider, 'value'))
        widgets.jslink((self.video_player, 'value'), (self.plot, 'time'))
        display(widgets.HBox([self.video_player, previous_frame_button, frame_slider, next_frame_button]))

    def __update_to_next_frame(self, button) -> None:
        if self.video_player.max > self.video_player.value:
            self.video_player.value += 1

    def __update_to_previous_frame(self, button) -> None:
        if self.video_player.min < self.video_player.value:
            self.video_player.value -= 1


class SkeletonColorUpdater:
    def __init__(self, skeleton: DrawableSkeleton, left_color_picker: ColorPicker,
                 right_color_picker: ColorPicker, center_color_picker: ColorPicker):
        self.skeleton: DrawableSkeleton = skeleton
        self.left_color_picker = left_color_picker
        self.right_color_picker = right_color_picker
        self.center_color_picker = center_color_picker

    def update_left_objects(self, current_color):
        for obj in self.skeleton.get_left_objects():
            obj.color = self.__get_as_hex_int(current_color.new)

    def update_right_objects(self, current_color):
        for obj in self.skeleton.get_right_objects():
            obj.color = self.__get_as_hex_int(current_color.new)

    def update_center_objects(self, current_color):
        for obj in self.skeleton.get_center_objects():
            obj.color = self.__get_as_hex_int(current_color.new)

    def update_all_objects(self, current_color):
        new_color = current_color.new
        for obj in self.skeleton:
            obj.color = self.__get_as_hex_int(new_color)
        self.left_color_picker.value = new_color
        self.right_color_picker.value = new_color
        self.center_color_picker.value = new_color

    @staticmethod
    def __get_as_hex_int(color: str):
        return int(color.replace('#', '0x'), 16)


class ColorPickerSynchronizer:
    def __init__(self, left_skeleton_color_pickers: List[ColorPicker], right_skeleton_color_pickers: List[ColorPicker],
                 center_skeleton_color_pickers: List[ColorPicker], all_skeleton_color_pickers: List[ColorPicker],
                 left_synchronizer_color_picker: ColorPicker, right_synchronizer_color_picker: ColorPicker,
                 center_synchronizer_color_picker: ColorPicker):
        self.left_skeleton_color_pickers: List[ColorPicker] = left_skeleton_color_pickers
        self.right_skeleton_color_pickers: List[ColorPicker] = right_skeleton_color_pickers
        self.center_skeleton_color_pickers: List[ColorPicker] = center_skeleton_color_pickers
        self.all_skeleton_color_pickers: List[ColorPicker] = all_skeleton_color_pickers
        self.left_synchronizer_color_picker: ColorPicker = left_synchronizer_color_picker
        self.right_synchronizer_color_picker: ColorPicker = right_synchronizer_color_picker
        self.center_synchronizer_color_picker: ColorPicker = center_synchronizer_color_picker

    def sync_left_color_pickers(self, current_color):
        for color_picker in self.left_skeleton_color_pickers:
            color_picker.value = current_color.new

    def sync_right_color_pickers(self, current_color):
        for color_picker in self.right_skeleton_color_pickers:
            color_picker.value = current_color.new

    def sync_center_color_pickers(self, current_color):
        for color_picker in self.center_skeleton_color_pickers:
            color_picker.value = current_color.new

    def sync_all_color_pickers(self, current_color):
        for color_picker in self.all_skeleton_color_pickers:
            color_picker.value = current_color.new
        self.left_synchronizer_color_picker.value = current_color.new
        self.right_synchronizer_color_picker.value = current_color.new
        self.center_synchronizer_color_picker.value = current_color.new


class ColorChanger:
    def __init__(self):
        super(ColorChanger, self).__init__()

    def display_color_changer(self, skeletons: List[DrawableSkeleton]):
        skeleton_color_changer_tabs: List[VBox] = []
        pred_color_pickers: List[Tuple[ColorPicker, ColorPicker, ColorPicker, ColorPicker]] = []
        gt_color_pickers: List[Tuple[ColorPicker, ColorPicker, ColorPicker, ColorPicker]] = []
        for skeleton in skeletons:
            skeleton_color_changer_tab, color_pickers = self.__create_color_changer_tab(skeleton)
            skeleton_color_changer_tabs.append(skeleton_color_changer_tab)
            if skeleton.is_ground_truth:
                gt_color_pickers.append(color_pickers)
            else:
                pred_color_pickers.append(color_pickers)
        if len(pred_color_pickers) != 0 and len(gt_color_pickers) != 0:
            pred_skeleton_color_changer_tab = self.__create_color_synchronizer_tab(pred_color_pickers)
            gt_skeleton_color_changer_tab = self.__create_color_synchronizer_tab(gt_color_pickers)
            skeleton_color_changer_tabs.append(pred_skeleton_color_changer_tab)
            skeleton_color_changer_tabs.append(gt_skeleton_color_changer_tab)
        color_changer_widget: Tab = Tab(children=skeleton_color_changer_tabs)
        for i in range(len(skeletons)):
            color_changer_widget.set_title(i, 'Skeleton {:d} colors'.format(i + 1))
        if len(skeleton_color_changer_tabs) > len(skeletons):
            color_changer_widget.set_title(len(skeletons), 'Pred skeleton colors')
            color_changer_widget.set_title(len(skeletons) + 1, 'GT skeleton colors')
        display(color_changer_widget)

    def __create_color_changer_tab(self, skeleton: DrawableSkeleton) ->\
            Tuple[VBox, Tuple[ColorPicker, ColorPicker, ColorPicker, ColorPicker]]:
        left_color_picker: ColorPicker = ColorPicker(
            description='Left color:',
            value=self.__get_as_html_color(skeleton.get_left_objects()[0].color))
        right_color_picker: ColorPicker = ColorPicker(
            description='Right color:',
            value=self.__get_as_html_color(skeleton.get_right_objects()[0].color))
        center_color_picker: ColorPicker = ColorPicker(
            description='Center color:',
            value=self.__get_as_html_color(skeleton.get_center_objects()[0].color))
        all_color_picker: ColorPicker = ColorPicker(description='All color:', value='#ffffff')
        skeleton_color_updater: SkeletonColorUpdater = SkeletonColorUpdater(
            skeleton, left_color_picker, right_color_picker, center_color_picker)
        left_color_picker.observe(skeleton_color_updater.update_left_objects, names='value')
        right_color_picker.observe(skeleton_color_updater.update_right_objects, names='value')
        center_color_picker.observe(skeleton_color_updater.update_center_objects, names='value')
        all_color_picker.observe(skeleton_color_updater.update_all_objects, names='value')
        return VBox([left_color_picker, right_color_picker, center_color_picker, all_color_picker]),\
            (left_color_picker, right_color_picker, center_color_picker, all_color_picker)

    @staticmethod
    def __create_color_synchronizer_tab(
            color_pickers: List[Tuple[ColorPicker, ColorPicker, ColorPicker, ColorPicker]]) -> VBox:
        left_color_picker: ColorPicker = ColorPicker(description='Left color:', value='#ffffff')
        right_color_picker: ColorPicker = ColorPicker(description='Right color:', value='#ffffff')
        center_color_picker: ColorPicker = ColorPicker(description='Center color:', value='#ffffff')
        all_color_picker: ColorPicker = ColorPicker(description='All color:', value='#ffffff')
        color_picker_synchronizer: ColorPickerSynchronizer = ColorPickerSynchronizer(
            left_skeleton_color_pickers=[color_picker[0] for color_picker in color_pickers],
            right_skeleton_color_pickers=[color_picker[1] for color_picker in color_pickers],
            center_skeleton_color_pickers=[color_picker[2] for color_picker in color_pickers],
            all_skeleton_color_pickers=[color_picker[3] for color_picker in color_pickers],
            left_synchronizer_color_picker=left_color_picker,
            right_synchronizer_color_picker=right_color_picker,
            center_synchronizer_color_picker=center_color_picker)
        left_color_picker.observe(color_picker_synchronizer.sync_left_color_pickers, names='value')
        right_color_picker.observe(color_picker_synchronizer.sync_right_color_pickers, names='value')
        center_color_picker.observe(color_picker_synchronizer.sync_center_color_pickers, names='value')
        all_color_picker.observe(color_picker_synchronizer.sync_all_color_pickers, names='value')
        return VBox([left_color_picker, right_color_picker, center_color_picker, all_color_picker])

    @staticmethod
    def __get_as_html_color(color: int) -> str:
        return '{0:#0{1}x}'.format(color, 8).replace('0x', '#')


class SkeletonVisualizer:
    """ A 3D skeleton visualizer which can visualize skeletons in 3D plots."""

    def __init__(self, joint_set: JointSet, size_scalar: float = 1.0):
        self.joint_set: JointSet = joint_set
        self.plot: Optional[Plot] = None
        self.size_scalar: float = size_scalar
        self.skeletons: List[DrawableSkeleton] = []
        self.joint_names_visible: Checkbox = Checkbox(description="Show Joint Names")
        self.joint_coordinates_visible: Checkbox = Checkbox(description="Show Coordinates")

    def visualize(
            self, skeletons: np.ndarray, colors: List[Color] = None, include_names: bool = False,
            include_coordinates: bool = False, automatic_camera_orientation: bool = False,
            is_gt_list: List[bool] = None) -> None:
        self.__assert_include_arguments(include_names, include_coordinates)
        self.__assert_skeleton_shapes(skeletons)
        colors = self.__init_colors(skeletons.shape[0], colors)
        if include_names:
            skeleton_converter = Skeleton.to_drawable_skeleton_with_names
        elif include_coordinates:
            skeleton_converter = Skeleton.to_drawable_skeleton_with_coordinates
        else:
            skeleton_converter = Skeleton.to_drawable_skeleton
        self.__create_skeleton_plot(skeletons=skeletons, skeleton_converter=skeleton_converter, colors=colors,
                                    automatic_camera_orientation=automatic_camera_orientation, is_gt_list=is_gt_list)
        self.plot.display()
        self.__display_checkboxes(include_names, include_coordinates)
        color_changer: ColorChanger = ColorChanger()
        color_changer.display_color_changer(self.skeletons)

    def visualize_with_ground_truths(
            self, pred_skeletons: np.ndarray, gt_skeletons: np.ndarray,
            pred_colors: List[Color] = None, gt_colors: List[Color] = None,
            include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False) -> None:
        assert pred_skeletons.shape == gt_skeletons.shape, \
            'The predicate and ground truth skeleton arrays must have the same shape.'
        pred_colors = self.__init_colors(pred_skeletons.shape[0], pred_colors)
        gt_colors = self.__init_colors(gt_skeletons.shape[0], gt_colors)
        skeletons: np.ndarray = np.concatenate((pred_skeletons, gt_skeletons), axis=0)
        colors: List[Color] = pred_colors + gt_colors
        is_gt_list: List[bool] = [False] * len(pred_skeletons) + [True] * len(gt_skeletons)
        self.visualize(
            skeletons, colors, include_names, include_coordinates, automatic_camera_orientation, is_gt_list)

    def visualize_video(
            self, frames: np.ndarray, colors: List[Color] = None, fps: int = 15,
            include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False, is_gt_list: List[bool] = None) -> None:
        self.__assert_include_arguments(include_names, include_coordinates)
        assert len(frames.shape) == 4
        first_frame: np.ndarray = frames[0]
        self.__assert_skeleton_shapes(first_frame)
        colors = self.__init_colors(first_frame.shape[0], colors)
        skeleton_timestamps: List[Dict[str, np.ndarray]] = self.__get_skeleton_positions_timestamps(frames)
        if include_names:
            skeleton_converter = Skeleton.to_drawable_skeleton_for_video_with_names
        elif include_coordinates:
            skeleton_converter = Skeleton.to_drawable_skeleton_for_video_with_coordinates
        else:
            skeleton_converter = Skeleton.to_drawable_skeleton_for_video
        self.__create_skeleton_plot(skeletons=first_frame, skeleton_converter=skeleton_converter, colors=colors,
                                    positions=skeleton_timestamps,
                                    automatic_camera_orientation=automatic_camera_orientation, is_gt_list=is_gt_list)
        self.plot.display()
        video_player: VideoPlayer = VideoPlayer(self.plot)
        video_player.display_video_player(fps, frames)
        self.__display_checkboxes(include_names, include_coordinates)
        color_changer: ColorChanger = ColorChanger()
        color_changer.display_color_changer(self.skeletons)

    def visualize_video_from_file(
            self, file_name: str, colors: List[Color] = None, fps: int = 15,
            include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False) -> None:
        file: IO = open(file_name, 'rb')
        frames = pickle.load(file)
        file.close()
        self.visualize_video(
            frames, colors, fps, include_names, include_coordinates, automatic_camera_orientation)

    def visualize_video_with_ground_truths(
            self, pred_frames: np.ndarray, gt_frames: np.ndarray,
            pred_colors: List[Color] = None, gt_colors: List[Color] = None,
            fps: int = 15, include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False) -> None:
        pred_colors = self.__init_colors(pred_frames.shape[1], pred_colors)
        gt_colors = self.__init_colors(gt_frames.shape[1], gt_colors)
        frames: np.ndarray = np.concatenate((pred_frames, gt_frames), axis=1)
        colors: List[Color] = pred_colors + gt_colors
        is_gt_list: List[bool] = [False] * len(pred_colors) + [True] * len(gt_colors)
        self.visualize_video(
            frames, colors, fps, include_names, include_coordinates, automatic_camera_orientation, is_gt_list)

    def visualize_video_from_file_with_ground_truths(
            self, pred_file_name: str, gt_file_name: str,
            pred_colors: List[Color] = None, gt_colors: List[Color] = None,
            fps: int = 15, include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False) -> None:
        file: IO = open(pred_file_name, 'rb')
        pred_frames = pickle.load(file)
        file.close()
        file = open(gt_file_name, 'rb')
        gt_frames = pickle.load(file)
        file.close()
        self.visualize_video_with_ground_truths(
            pred_frames, gt_frames, pred_colors, gt_colors, fps,
            include_names, include_coordinates, automatic_camera_orientation)

    @staticmethod
    def __init_colors(number_of_skeletons: int, colors: List[Color]) -> List[Color]:
        if colors is None:
            colors = ['default'] * number_of_skeletons
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
    def __assert_include_arguments(include_names: bool, include_coordinates: bool) -> None:
        if include_names is True and include_coordinates is True:
            raise AttributeError('Either names or coordinates can be showed, but not both.')

    @staticmethod
    def __get_skeleton_positions_timestamps(frames: np.ndarray) -> List[Dict[str, np.ndarray]]:
        frames_swapped: np.ndarray = np.swapaxes(frames, 0, 1)
        return [{
            str(timestamp_index): frames_swapped[current_frame_index][timestamp_index]
            for timestamp_index in range(len(frames_swapped[current_frame_index]))
        } for current_frame_index in range(len(frames_swapped))]

    def __create_skeleton_plot(self, skeletons: np.ndarray, skeleton_converter: Callable[[Skeleton], DrawableSkeleton],
                               colors: List[Color], positions: Optional[Positions] = None,
                               automatic_camera_orientation: bool = False, is_gt_list: List[bool] = None) -> None:
        self.__init_skeleton_plot(skeletons, automatic_camera_orientation)
        skeleton_part_size: float = self.__calculate_skeleton_part_size(skeletons)
        if positions is None:
            positions = skeletons
        skeletons: List[Skeleton] = self.__get_skeletons(positions, colors, skeleton_part_size, is_gt_list)
        drawable_skeletons: List[DrawableSkeleton] = list(map(skeleton_converter, skeletons))
        self.__add_skeletons_to_plot(drawable_skeletons)

    def __init_skeleton_plot(self, skeletons: np.ndarray, automatic_camera_orientation: bool = False) -> None:
        self.plot = k3d.plot(antialias=1, camera_auto_fit=False)
        centroid: np.ndarray = np.average(np.average(skeletons, axis=0), axis=0)
        if automatic_camera_orientation:
            camera_up_vector: np.ndarray = np.zeros(shape=(3,))
            for line_indices in self.joint_set.vertically_aligned_line_indices:
                camera_up_vector += np.sum(
                    skeletons[:, line_indices[1]] - skeletons[:, line_indices[0]], axis=0)
            camera_up_vector /= np.linalg.norm(camera_up_vector, ord=2)
            self.plot.camera = [0.0, 0.0, 0.0,  # Camera position
                                centroid[0], centroid[1], centroid[2],  # Camera looking at
                                camera_up_vector[0], camera_up_vector[1], camera_up_vector[2]]  # Camera up vector
        else:
            self.plot.camera = [0.0, 0.0, 0.0,  # Camera position
                                0.0, 0.0, centroid[2],  # Camera looking at
                                0.0, -1.0, 0.0]  # Camera up vector

    def __calculate_skeleton_part_size(self, skeletons: np.ndarray) -> float:
        max_values = [abs(skeleton).max() for skeleton in skeletons]
        return (min(max_values) / 100.0) * self.size_scalar

    def __get_skeletons(self, positions: List[Positions], colors: List[Color],
                        skeleton_part_size: float, is_gt_list: List[bool] = None) -> List[Skeleton]:
        if is_gt_list is None:
            is_gt_list = [False] * len(colors)
        position_index, color_index, ground_truth_index = 0, 1, 2
        return list(map(lambda parameter_tuple: Skeleton(
            joint_positions=parameter_tuple[position_index], joint_set=self.joint_set,
            part_size=skeleton_part_size, color=parameter_tuple[color_index],
            is_ground_truth=parameter_tuple[ground_truth_index]
        ), zip(positions, colors, is_gt_list)))

    def __add_skeletons_to_plot(self, skeletons: List[DrawableSkeleton]) -> None:
        for drawable_skeleton in skeletons:
            self.plot += drawable_skeleton
            self.skeletons.append(drawable_skeleton)

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
