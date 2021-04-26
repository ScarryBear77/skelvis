import pickle
from typing import Optional, List, Dict, Callable, IO, Tuple

import ipywidgets as widgets
import k3d
import numpy as np
from IPython.display import display
from ipywidgets import Play, Checkbox, IntSlider, Button, ColorPicker, VBox, Tab, HBox, Accordion, Widget, Label
from k3d.plot import Plot

from .jointset import JointSet, MuPoTSJoints, OpenPoseJoints, CocoExJoints, PanopticJoints, Common14Joints
from .loss import L2
from .objects import Skeleton, DrawableSkeleton, Color, Positions


def create_video_player(fps: int, number_of_frames: int) -> Play:
    return Play(value=0, min=0, max=number_of_frames, step=1,
                interval=1000 / fps, description='Press play', disabled=False)


class VideoPlayer:
    def __init__(self, plot: Plot, fps: int, frames: np.ndarray, video_player: Play = None):
        self.plot: Plot = plot
        self.video_player: Optional[Play] = \
            video_player if video_player is not None else create_video_player(fps, number_of_frames=frames.shape[0] - 1)
        self.fps: int = fps
        self.frames: np.ndarray = frames

    def get_video_player_widget(self) -> HBox:
        frame_slider = IntSlider(value=self.video_player.value, min=self.video_player.min,
                                 max=self.video_player.max, step=self.video_player.step, description='Frame')
        next_frame_button = Button(description='Next frame')
        next_frame_button.on_click(self.__update_to_next_frame)
        previous_frame_button = Button(description='Previous frame')
        previous_frame_button.on_click(self.__update_to_previous_frame)
        widgets.jslink((self.video_player, 'value'), (frame_slider, 'value'))
        widgets.jslink((self.video_player, 'value'), (self.plot, 'time'))
        return HBox([self.video_player, previous_frame_button, frame_slider, next_frame_button])

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

    def get_color_changer_widget(self, skeletons: List[DrawableSkeleton]) -> Tab:
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
        return color_changer_widget

    def __create_color_changer_tab(self, skeleton: DrawableSkeleton) -> \
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
        return (VBox([left_color_picker, right_color_picker, center_color_picker, all_color_picker]),
                (left_color_picker, right_color_picker, center_color_picker, all_color_picker))

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


class LossContainer:
    def __init__(self, pred_skeletons: np.ndarray, gt_skeletons: np.ndarray, joint_set: JointSet,
                 loss: Callable[[np.ndarray, np.ndarray], np.ndarray], video_player: Optional[Play] = None):
        if video_player is not None:
            self.is_video: bool = True
            self.pred_skeletons: np.ndarray = np.swapaxes(pred_skeletons, 0, 1)
            self.gt_skeletons: np.ndarray = np.swapaxes(gt_skeletons, 0, 1)
            # Fields related to playing videos
            self.video_player = video_player
            self.min_loss_frame_index: int = 0
            self.max_loss_frame_index: int = 0
        else:
            self.is_video: bool = False
            self.pred_skeletons: np.ndarray = np.swapaxes(pred_skeletons[np.newaxis], 0, 1)
            self.gt_skeletons: np.ndarray = np.swapaxes(gt_skeletons[np.newaxis], 0, 1)
        self.joint_set: JointSet = joint_set
        self.number_of_skeletons: int = self.pred_skeletons.shape[0]
        # Joint loss related fields
        self.joint_losses: np.ndarray = loss(self.pred_skeletons, self.gt_skeletons)
        self.min_joint_loss_indices = np.argmin(self.joint_losses, axis=-1)
        self.max_joint_loss_indices = np.argmax(self.joint_losses, axis=-1)
        # Joint loss related labels
        self.joint_loss_labels: List[List[Label]] = []
        self.all_joint_loss_labels: List[Label] = [Label(value='test') for _ in range(self.number_of_skeletons)]
        self.min_joint_loss_name_labels: List[Label] = [Label(value='test') for _ in range(self.number_of_skeletons)]
        self.max_joint_loss_name_labels: List[Label] = [Label(value='test') for _ in range(self.number_of_skeletons)]
        # Skeleton loss related fields
        self.skeleton_losses = np.sum(self.joint_losses, axis=-1)
        self.all_losses = np.sum(self.skeleton_losses, axis=0)
        self.min_skeleton_loss_indices = np.argmin(self.skeleton_losses, axis=0) + 1
        self.max_skeleton_loss_indices = np.argmax(self.skeleton_losses, axis=0) + 1
        # Skeleton loss related labels
        self.skeleton_loss_labels: List[Label] = [Label(value='test') for _ in range(self.number_of_skeletons)]
        self.all_loss_label: Label = Label(value='test')
        self.min_loss_index_label: Label = Label(value='test')
        self.max_loss_index_label: Label = Label(value='test')

    def get_loss_tab(self) -> Tab:
        loss_tab: Tab = self.__create_empty_loss_tabs()
        self.__set_loss_labels(0)
        if self.is_video:
            self.video_player.observe(self.__update_losses, names='value')
        return loss_tab

    def __create_empty_loss_tabs(self) -> Tab:
        skeleton_loss_tabs: List[HBox] = [self.__create_loss_tab_for_skeleton(i)
                                          for i in range(self.number_of_skeletons)]
        all_losses_tab: HBox = self.__create_all_losses_tab()
        loss_tabs: List[HBox] = [all_losses_tab] + skeleton_loss_tabs
        loss_tab: Tab = Tab(children=loss_tabs)
        loss_tab.set_title(0, "All losses")
        for i in range(len(loss_tabs) - 1):
            loss_tab.set_title(i + 1, 'Skeleton {:d} losses'.format(i + 1))
        return loss_tab

    def __create_loss_tab_for_skeleton(self, index: int) -> HBox:
        joint_loss_labels: List[Label] = [Label(value='test') for _ in range(self.joint_set.number_of_joints)]
        joint_names: VBox = VBox(children=[Label(value=name) for name in self.joint_set.names])
        joint_losses: VBox = VBox(children=joint_loss_labels)
        self.joint_loss_labels.append(joint_loss_labels)
        statistics_labels: VBox = VBox(
            children=[Label(value='All losses:'),
                      Label(value='Joint with min loss:'),
                      Label(value='Joint with max loss:')])
        skeleton_values: VBox = VBox(
            children=[self.all_joint_loss_labels[index],
                      self.min_joint_loss_name_labels[index],
                      self.max_joint_loss_name_labels[index]])
        return HBox(children=[joint_names, joint_losses, statistics_labels, skeleton_values])

    def __create_all_losses_tab(self) -> HBox:
        skeleton_labels: VBox = VBox(
            children=[Label(value='Skeleton {:d} loss:'.format(i + 1)) for i in range(self.number_of_skeletons)])
        skeleton_losses: VBox = VBox(children=self.skeleton_loss_labels)
        statistics_labels_column: List[Label] = [Label(value='All losses:'),
                                                 Label(value='Min loss skeleton index:'),
                                                 Label(value='Max loss skeleton index:')]
        statistics_values_column: List[Label] = [self.all_loss_label,
                                                 self.min_loss_index_label,
                                                 self.max_loss_index_label]
        jump_buttons: List[Button] = self.__create_frame_jump_buttons(statistics_labels_column,
                                                                      statistics_values_column)
        statistics_labels: VBox = VBox(children=statistics_labels_column)
        statistics_values: VBox = VBox(children=statistics_values_column)
        all_losses_tab: HBox = HBox(
            children=[skeleton_labels, skeleton_losses,
                      statistics_labels, statistics_values
                      ] + ([VBox(children=jump_buttons)] if len(jump_buttons) > 0 else [])
        )
        return all_losses_tab

    def __create_frame_jump_buttons(self, statistics_labels_column: List[Label],
                                    statistics_values_column: List[Label]) -> List[Button]:
        jump_buttons: List[Button] = []
        if self.is_video:
            self.min_loss_frame_index: int = np.argmin(self.all_losses)
            min_loss_jump_button: Button = Button(description='Min loss frame')
            min_loss_jump_button.on_click(self.__jump_to_min_loss_frame)
            self.max_loss_frame_index: int = np.argmax(self.all_losses)
            max_loss_jump_button: Button = Button(description='Max loss frame')
            max_loss_jump_button.on_click(self.__jump_to_max_loss_frame)
            statistics_labels_column.append(Label(value='Min loss frame index:'))
            statistics_labels_column.append(Label(value='Max loss frame index:'))
            statistics_values_column.append(Label(value=str(self.min_loss_frame_index)))
            statistics_values_column.append(Label(value=str(self.max_loss_frame_index)))
            jump_buttons.append(min_loss_jump_button)
            jump_buttons.append(max_loss_jump_button)
        return jump_buttons

    def __jump_to_min_loss_frame(self, button):
        self.video_player.value = self.min_loss_frame_index

    def __jump_to_max_loss_frame(self, button):
        self.video_player.value = self.max_loss_frame_index

    def __set_loss_labels(self, index: int) -> None:
        for i in range(self.number_of_skeletons):
            self.skeleton_loss_labels[i].value = '{:.3f}'.format(self.skeleton_losses[i][index])
            for j in range(self.joint_set.number_of_joints):
                self.joint_loss_labels[i][j].value = '{:.3f}'.format(self.joint_losses[i][index][j])
            self.all_joint_loss_labels[i].value = '{:.3f}'.format(self.skeleton_losses[i][index])
            self.min_joint_loss_name_labels[i].value = self.joint_set.names[self.min_joint_loss_indices[i][index]]
            self.max_joint_loss_name_labels[i].value = self.joint_set.names[self.max_joint_loss_indices[i][index]]
        self.all_loss_label.value = '{:.3f}'.format(self.all_losses[index])
        self.min_loss_index_label.value = str(self.min_skeleton_loss_indices[index])
        self.max_loss_index_label.value = str(self.max_skeleton_loss_indices[index])

    def __update_losses(self, current_frame):
        self.__set_loss_labels(current_frame.new)


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
            is_gt_list: List[bool] = None, additional_tabs: List[Tuple[str, Widget]] = None) -> None:
        self.__assert_include_arguments(include_names, include_coordinates)
        self.__assert_skeleton_shapes(skeletons)
        colors = self.__init_colors(skeletons.shape[0], colors)
        if include_names:
            skeleton_converter = Skeleton.to_drawable_skeleton_with_names
        elif include_coordinates:
            skeleton_converter = Skeleton.to_drawable_skeleton_with_coordinates
        else:
            skeleton_converter = Skeleton.to_drawable_skeleton
        self.__create_skeleton_plot(
            skeletons=skeletons, skeleton_converter=skeleton_converter, colors=colors,
            automatic_camera_orientation=automatic_camera_orientation, is_gt_list=is_gt_list)
        self.plot.display()
        self.__link_text_widgets(include_names, include_coordinates)
        visibility_widget: HBox = HBox([self.joint_names_visible, self.joint_coordinates_visible])
        color_changer: ColorChanger = ColorChanger()
        color_changer_tab: Tab = color_changer.get_color_changer_widget(self.skeletons)
        self.__display_interface([('Change visibilities', visibility_widget),
                                  ('Change colors', color_changer_tab)] +
                                 ([] if additional_tabs is None else additional_tabs))

    def visualize_from_file(
            self, file_name: str, colors: List[Color] = None, include_names: bool = False,
            include_coordinates: bool = False, automatic_camera_orientation: bool = False,
            is_gt_list: List[bool] = None, additional_tabs: List[Tuple[str, Widget]] = None) -> None:
        file: IO = open(file_name, 'rb')
        skeletons = pickle.load(file)
        file.close()
        self.visualize(skeletons, colors, include_names, include_coordinates,
                       automatic_camera_orientation, is_gt_list,  additional_tabs)

    def visualize_with_ground_truths(
            self, pred_skeletons: np.ndarray, gt_skeletons: np.ndarray,
            pred_colors: List[Color] = None, gt_colors: List[Color] = None,
            include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False, include_losses: bool = True,
            loss: Callable[[np.ndarray, np.ndarray], np.ndarray] = L2) -> None:
        assert pred_skeletons.shape == gt_skeletons.shape, \
            'The predicate and ground truth skeleton arrays must have the same shape.'
        pred_colors = self.__init_colors(pred_skeletons.shape[0], pred_colors)
        gt_colors = self.__init_colors(gt_skeletons.shape[0], gt_colors)
        skeletons: np.ndarray = np.concatenate((pred_skeletons, gt_skeletons), axis=0)
        colors: List[Color] = pred_colors + gt_colors
        is_gt_list: List[bool] = [False] * len(pred_skeletons) + [True] * len(gt_skeletons)
        if include_losses:
            loss_container: LossContainer = LossContainer(pred_skeletons, gt_skeletons, self.joint_set, loss)
        self.visualize(
            skeletons, colors, include_names, include_coordinates, automatic_camera_orientation,
            is_gt_list, additional_tabs=[('Losses', loss_container.get_loss_tab())] if include_losses else None)

    def visualize_from_file_with_ground_truths(
            self, pred_file_name: str, gt_file_name: str,
            pred_colors: List[Color] = None, gt_colors: List[Color] = None,
            include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False, include_losses: bool = True,
            loss: Callable[[np.ndarray, np.ndarray], np.ndarray] = L2) -> None:
        file: IO = open(pred_file_name, 'rb')
        pred_skeletons = pickle.load(file)
        file.close()
        file = open(gt_file_name, 'rb')
        gt_skeletons = pickle.load(file)
        file.close()
        self.visualize_with_ground_truths(
            pred_skeletons, gt_skeletons, pred_colors, gt_colors, include_names,
            include_coordinates, automatic_camera_orientation, include_losses, loss)

    def visualize_video(
            self, frames: np.ndarray, colors: List[Color] = None, fps: int = 15,
            include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False, is_gt_list: List[bool] = None,
            player: Optional[Play] = None, additional_tabs: List[Tuple[str, Widget]] = None) -> None:
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
        self.__create_skeleton_plot(
            skeletons=first_frame, skeleton_converter=skeleton_converter, colors=colors, positions=skeleton_timestamps,
            automatic_camera_orientation=automatic_camera_orientation, is_gt_list=is_gt_list)
        self.__link_text_widgets(include_names, include_coordinates)
        self.plot.display()
        visibility_widget: HBox = HBox([self.joint_names_visible, self.joint_coordinates_visible])
        video_player: VideoPlayer = VideoPlayer(self.plot, fps, frames, player)
        video_player_widget: HBox = video_player.get_video_player_widget()
        color_changer: ColorChanger = ColorChanger()
        color_changer_tab: Tab = color_changer.get_color_changer_widget(self.skeletons)
        self.__display_interface([('Play video', video_player_widget),
                                  ('Change visibilities', visibility_widget),
                                  ('Change colors', color_changer_tab)] +
                                 ([] if additional_tabs is None else additional_tabs))

    def visualize_video_from_file(
            self, file_name: str, colors: List[Color] = None, fps: int = 15,
            include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False, is_gt_list: List[bool] = None,
            player: Optional[Play] = None, additional_tabs: List[Tuple[str, Widget]] = None) -> None:
        file: IO = open(file_name, 'rb')
        frames = pickle.load(file)
        file.close()
        self.visualize_video(
            frames, colors, fps, include_names, include_coordinates,
            automatic_camera_orientation, is_gt_list, player, additional_tabs)

    def visualize_video_with_ground_truths(
            self, pred_frames: np.ndarray, gt_frames: np.ndarray,
            pred_colors: List[Color] = None, gt_colors: List[Color] = None,
            fps: int = 15, include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False, include_losses: bool = True,
            loss: Callable[[np.ndarray, np.ndarray], np.ndarray] = L2) -> None:
        pred_colors = self.__init_colors(pred_frames.shape[1], pred_colors)
        gt_colors = self.__init_colors(gt_frames.shape[1], gt_colors)
        frames: np.ndarray = np.concatenate((pred_frames, gt_frames), axis=1)
        colors: List[Color] = pred_colors + gt_colors
        is_gt_list: List[bool] = [False] * len(pred_colors) + [True] * len(gt_colors)
        player: Play = create_video_player(fps, frames.shape[0] - 1)
        if include_losses:
            loss_container: LossContainer = LossContainer(pred_frames, gt_frames, self.joint_set, loss, player)
        self.visualize_video(
            frames, colors, fps, include_names, include_coordinates, automatic_camera_orientation,
            is_gt_list, player, additional_tabs=[('Losses', loss_container.get_loss_tab())] if include_losses else None)

    def visualize_video_from_file_with_ground_truths(
            self, pred_file_name: str, gt_file_name: str,
            pred_colors: List[Color] = None, gt_colors: List[Color] = None,
            fps: int = 15, include_names: bool = False, include_coordinates: bool = False,
            automatic_camera_orientation: bool = False, include_losses: bool = True,
            loss: Callable[[np.ndarray, np.ndarray], np.ndarray] = L2) -> None:
        file: IO = open(pred_file_name, 'rb')
        pred_frames = pickle.load(file)
        file.close()
        file = open(gt_file_name, 'rb')
        gt_frames = pickle.load(file)
        file.close()
        self.visualize_video_with_ground_truths(
            pred_frames, gt_frames, pred_colors, gt_colors, fps,
            include_names, include_coordinates, automatic_camera_orientation, include_losses, loss)

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

    def __link_text_widgets(self, include_names: bool, include_coordinates: bool) -> None:
        if include_names:
            self.__link_joint_name_visibility_with_checkbox()
        if include_coordinates:
            self.__link_joint_coordinate_visibility_with_checkbox()

    def __link_joint_name_visibility_with_checkbox(self) -> None:
        for skeleton in self.skeletons:
            for joint_name in skeleton.joint_names:
                widgets.jslink((joint_name, 'visible'), (self.joint_names_visible, 'value'))

    def __link_joint_coordinate_visibility_with_checkbox(self) -> None:
        for skeleton in self.skeletons:
            for joint_coordinate in skeleton.joint_coordinates:
                widgets.jslink((joint_coordinate, 'visible'), (self.joint_coordinates_visible, 'value'))

    @staticmethod
    def __display_interface(widget_tuples: List[Tuple[str, Widget]]) -> None:
        interface: Accordion = Accordion(children=list(map(lambda widget_tuple: widget_tuple[1], widget_tuples)))
        for i in range(len(widget_tuples)):
            interface.set_title(i, widget_tuples[i][0])
        display(interface)


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
