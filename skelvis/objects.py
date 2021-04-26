from typing import Union, NewType, List, Dict, Final, Tuple

import k3d
import numpy as np
from k3d.objects import Group, Line, Points, Text, Drawable

from .jointset import JointSet

Color = NewType('Color', Union[str, int])
PositionTimestamps = NewType('PositionTimestamps', Dict[str, np.ndarray])
Positions = NewType('Positions', Union[np.ndarray, PositionTimestamps])

DEFAULT_COLORS: Final[Dict[str, int]] = {
    'red': 0xFF0000, 'green': 0x00FF00, 'blue': 0x0000FF,
    'yellow': 0xFFFF00, 'teal': 0x00FFFF, 'purple': 0xFF00FF,
    'white': 0xFFFFFF, 'black': 0x000000
}
DEFAULT_TEXT_SIZE: Final[float] = 0.45
COORDINATE_FORMAT: Final[str] = '({:.2f}, {:.2f}, {:.2f})'


class DrawableSkeleton(Group):
    def __init__(self, joint_set: JointSet, joint_points: List[Points], joint_lines: List[Line],
                 is_ground_truth: bool, joint_names: List[Text] = None, joint_coordinates: List[Text] = None):
        drawable_objects: List[Drawable] = joint_points + joint_lines
        if joint_names is not None:
            drawable_objects += joint_names
        if joint_coordinates is not None:
            drawable_objects += joint_coordinates
        super().__init__(drawable_objects)
        self.joint_set: JointSet = joint_set
        self.joint_points: List[Points] = joint_points
        self.joint_lines: List[Line] = joint_lines
        self.is_ground_truth: bool = is_ground_truth
        self.joint_names: List[Text] = joint_names
        self.joint_coordinates: List[Text] = joint_coordinates

    def get_left_objects(self) -> List[Drawable]:
        return [self.joint_points[left_point_index]
                for left_point_index in self.joint_set.left_joint_indices] + \
               [self.joint_lines[left_line_index]
                for left_line_index in self.joint_set.left_line_indices]

    def get_right_objects(self) -> List[Drawable]:
        return [self.joint_points[right_point_index]
                for right_point_index in self.joint_set.right_joint_indices] + \
               [self.joint_lines[right_line_index]
                for right_line_index in self.joint_set.right_line_indices]

    def get_center_objects(self) -> List[Drawable]:
        return [self.joint_points[center_point_index]
                for center_point_index in self.joint_set.center_joint_indices] + \
               [self.joint_lines[center_line_index]
                for center_line_index in self.joint_set.center_line_indices]


class Skeleton:
    def __init__(self, joint_positions: Positions, joint_set: JointSet,
                 part_size: float, color: Color, is_ground_truth: bool):
        self.joint_positions: Positions = joint_positions
        self.joint_set: JointSet = joint_set
        self.part_size: float = part_size
        self.color: Color = color
        self.is_ground_truth: bool = is_ground_truth

    def to_drawable_skeleton(self) -> DrawableSkeleton:
        joint_points: List[Points] = self.__get_joint_points()
        joint_lines: List[Line] = self.__get_joint_lines()
        return DrawableSkeleton(self.joint_set, joint_points, joint_lines, self.is_ground_truth)

    def to_drawable_skeleton_with_names(self) -> DrawableSkeleton:
        joint_points: List[Points] = self.__get_joint_points()
        joint_lines: List[Line] = self.__get_joint_lines()
        joint_names: List[Text] = self.__get_joint_names()
        return DrawableSkeleton(self.joint_set, joint_points, joint_lines,
                                self.is_ground_truth, joint_names=joint_names)

    def to_drawable_skeleton_with_coordinates(self) -> DrawableSkeleton:
        joint_points: List[Points] = self.__get_joint_points()
        joint_lines: List[Line] = self.__get_joint_lines()
        joint_coordinates: List[Text] = self.__get_joint_coordinates()
        return DrawableSkeleton(self.joint_set, joint_points, joint_lines,
                                self.is_ground_truth, joint_coordinates=joint_coordinates)

    def to_drawable_skeleton_for_video(self) -> DrawableSkeleton:
        joint_points: List[Points] = self.__get_joint_points_for_video()
        joint_lines: List[Line] = self.__get_joint_lines_for_video()
        return DrawableSkeleton(self.joint_set, joint_points, joint_lines, self.is_ground_truth)

    def to_drawable_skeleton_for_video_with_names(self) -> DrawableSkeleton:
        joint_points: List[Points] = self.__get_joint_points_for_video()
        joint_lines: List[Line] = self.__get_joint_lines_for_video()
        joint_names: List[Text] = self.__get_joint_names_for_video()
        return DrawableSkeleton(self.joint_set, joint_points, joint_lines,
                                self.is_ground_truth, joint_names=joint_names)

    def to_drawable_skeleton_for_video_with_coordinates(self) -> DrawableSkeleton:
        joint_points: List[Points] = self.__get_joint_points_for_video()
        joint_lines: List[Line] = self.__get_joint_lines_for_video()
        joint_coordinates: List[Text] = self.__get_joint_coordinates_for_video()
        return DrawableSkeleton(self.joint_set, joint_points, joint_lines,
                                self.is_ground_truth, joint_coordinates=joint_coordinates)

    def __get_joint_points(self) -> List[Points]:
        skeleton_joint_colors: np.ndarray = self.__get_joint_colors()
        return [k3d.points(
            positions=self.joint_positions[line_index], point_size=self.part_size,
            shader='mesh', color=int(skeleton_joint_colors[line_index])
        ) for line_index in range(self.joint_set.number_of_joints)]

    def __get_joint_points_for_video(self) -> List[Points]:
        skeleton_joint_colors: np.ndarray = self.__get_joint_colors()
        return [k3d.points(
            positions={
                current_timestamp[0]: current_timestamp[1][line_index]
                for current_timestamp in self.joint_positions.items()
            }, color=int(skeleton_joint_colors[line_index]),
            point_size=self.part_size, shader='mesh'
        ) for line_index in range(self.joint_set.number_of_joints)]

    def __get_joint_lines(self) -> List[Line]:
        skeleton_line_colors: np.ndarray = self.__get_line_colors()
        return [k3d.line(
            vertices=[
                self.joint_positions[self.joint_set.limb_graph[line_index][0]],
                self.joint_positions[self.joint_set.limb_graph[line_index][1]]
            ], width=self.part_size / 2.2,
            color=int(skeleton_line_colors[line_index]),
            shader='mesh'
        ) for line_index in range(len(self.joint_set.limb_graph))]

    def __get_joint_lines_for_video(self) -> List[Line]:
        skeleton_line_colors: np.ndarray = self.__get_line_colors()
        return [k3d.line(
            vertices={
                current_timestamp[0]: np.array([
                    current_timestamp[1][self.joint_set.limb_graph[line_index][0]],
                    current_timestamp[1][self.joint_set.limb_graph[line_index][1]]
                ]) for current_timestamp in self.joint_positions.items()
            }, width=self.part_size / 2.2,
            color=int(skeleton_line_colors[line_index]),
            shader='mesh'
        ) for line_index in range(len(self.joint_set.limb_graph))]

    def __get_joint_names(self) -> List[Text]:
        return [k3d.text(
            text=self.joint_set.names[joint_index],
            position=self.joint_positions[joint_index],
            size=DEFAULT_TEXT_SIZE, label_box=False, color=DEFAULT_COLORS.get('black')
        ) for joint_index in range(self.joint_set.number_of_joints)]

    def __get_joint_names_for_video(self) -> List[Text]:
        return [k3d.text(
            text=self.joint_set.names[joint_index],
            position={
                current_timestamp[0]: current_timestamp[1][joint_index]
                for current_timestamp in self.joint_positions.items()
            }, size=DEFAULT_TEXT_SIZE, label_box=False, color=DEFAULT_COLORS.get('black')
        ) for joint_index in range(self.joint_set.number_of_joints)]

    def __get_joint_coordinates(self) -> List[Text]:
        return [k3d.text(
            text=COORDINATE_FORMAT.format(
                self.joint_positions[joint_index][0],
                self.joint_positions[joint_index][1],
                self.joint_positions[joint_index][2]
            ), position=self.joint_positions[joint_index],
            size=DEFAULT_TEXT_SIZE, label_box=False, color=DEFAULT_COLORS.get('black')
        ) for joint_index in range(self.joint_set.number_of_joints)]

    def __get_joint_coordinates_for_video(self) -> List[Text]:
        return [k3d.text(
            text={
                current_timestamp[0]: COORDINATE_FORMAT.format(
                    current_timestamp[1][joint_index][0],
                    current_timestamp[1][joint_index][1],
                    current_timestamp[1][joint_index][2])
                for current_timestamp in self.joint_positions.items()
            }, position={
                current_timestamp[0]: current_timestamp[1][joint_index]
                for current_timestamp in self.joint_positions.items()
            }, size=DEFAULT_TEXT_SIZE, label_box=False, color=DEFAULT_COLORS.get('black')
        ) for joint_index in range(self.joint_set.number_of_joints)]

    def __get_joint_colors(self) -> np.ndarray:
        joint_colors_shape: Tuple[int] = (self.joint_set.number_of_joints,)
        if self.color == 'default':
            joint_colors = np.zeros(shape=joint_colors_shape, dtype='uint32')
            if self.is_ground_truth:
                joint_colors[self.joint_set.left_joint_indices] = DEFAULT_COLORS.get('yellow')
                joint_colors[self.joint_set.right_joint_indices] = DEFAULT_COLORS.get('green')
                joint_colors[self.joint_set.center_joint_indices] = DEFAULT_COLORS.get('teal')
            else:
                joint_colors[self.joint_set.left_joint_indices] = DEFAULT_COLORS.get('red')
                joint_colors[self.joint_set.right_joint_indices] = DEFAULT_COLORS.get('blue')
                joint_colors[self.joint_set.center_joint_indices] = DEFAULT_COLORS.get('white')
            return joint_colors
        else:
            return np.full(
                shape=joint_colors_shape,
                fill_value=self.__get_color(self.color),
                dtype='uint32')

    def __get_line_colors(self) -> np.ndarray:
        line_colors_shape: Tuple[int] = (self.joint_set.number_of_joints - 1,)
        if self.color == 'default':
            line_colors = np.zeros(shape=line_colors_shape, dtype='uint32')
            if self.is_ground_truth:
                line_colors[self.joint_set.left_line_indices] = DEFAULT_COLORS.get('yellow')
                line_colors[self.joint_set.right_line_indices] = DEFAULT_COLORS.get('green')
                line_colors[self.joint_set.center_line_indices] = DEFAULT_COLORS.get('teal')
            else:
                line_colors[self.joint_set.left_line_indices] = DEFAULT_COLORS.get('red')
                line_colors[self.joint_set.right_line_indices] = DEFAULT_COLORS.get('blue')
                line_colors[self.joint_set.center_line_indices] = DEFAULT_COLORS.get('white')
            return line_colors
        else:
            return np.full(
                shape=line_colors_shape,
                fill_value=self.__get_color(self.color),
                dtype='uint32')

    @staticmethod
    def __get_color(color: Color) -> int:
        if isinstance(color, str):
            if color in DEFAULT_COLORS.keys():
                return DEFAULT_COLORS.get(color)
            else:
                raise KeyError('Default colors do not contain ' + color + '.')
        elif isinstance(color, int):
            return color
        else:
            raise TypeError('Color must be either of type \'int\' or \'str\'.')
