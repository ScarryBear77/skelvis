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
    def __init__(self, joint_points: Points, joint_lines: List[Line],
                 joint_names: List[Text] = None, joint_coordinates: List[Text] = None):
        points: List[Drawable] = [joint_points]
        drawable_objects: List[Drawable] = points + joint_lines
        if joint_names is not None:
            drawable_objects += joint_names
        if joint_coordinates is not None:
            drawable_objects += joint_coordinates
        super().__init__(drawable_objects)
        self.joint_points: Points = joint_points
        self.joint_lines: List[Line] = joint_lines
        self.joint_names: List[Text] = joint_names
        self.joint_coordinates: List[Text] = joint_coordinates


class Skeleton:
    def __init__(self, joint_positions: Positions,
                 joint_set: JointSet, part_size: float, color: Color):
        self.joint_positions: Positions = joint_positions
        self.joint_set: JointSet = joint_set
        self.part_size: float = part_size
        self.color: Color = color

    def to_drawable_skeleton(self) -> DrawableSkeleton:
        joint_points: Points = self.__get_joint_points()
        joint_lines: List[Line] = self.__get_joint_lines()
        return DrawableSkeleton(joint_points, joint_lines)

    def to_drawable_skeleton_with_coordinates(self) -> DrawableSkeleton:
        joint_points: Points = self.__get_joint_points()
        joint_lines: List[Line] = self.__get_joint_lines()
        joint_coordinates: List[Text] = self.__get_joint_coordinates()
        return DrawableSkeleton(joint_points, joint_lines, joint_coordinates=joint_coordinates)

    def to_drawable_skeleton_with_names(self) -> DrawableSkeleton:
        joint_points: Points = self.__get_joint_points()
        joint_lines: List[Line] = self.__get_joint_lines()
        joint_names: List[Text] = self.__get_joint_names()
        return DrawableSkeleton(joint_points, joint_lines, joint_names=joint_names)

    def to_drawable_skeleton_for_video(self) -> DrawableSkeleton:
        joint_points: Points = self.__get_joint_points()
        joint_lines: List[Line] = self.__get_joint_lines_for_video()
        return DrawableSkeleton(joint_points, joint_lines)

    def to_drawable_skeleton_for_video_with_coordinates(self) -> DrawableSkeleton:
        joint_points: Points = self.__get_joint_points()
        joint_lines: List[Line] = self.__get_joint_lines_for_video()
        joint_coordinates: List[Text] = self.__get_joint_coordinates_for_video()
        return DrawableSkeleton(joint_points, joint_lines, joint_coordinates=joint_coordinates)

    def __get_joint_points(self) -> Points:
        skeleton_joint_colors: np.ndarray = self.__get_joint_colors()
        return k3d.points(
            positions=self.joint_positions, point_size=self.part_size,
            shader='mesh', colors=skeleton_joint_colors
        )

    def __get_joint_lines(self) -> List[Line]:
        skeleton_line_colors: np.ndarray = self.__get_line_colors()
        return [k3d.line(
            vertices=[
                self.joint_positions[self.joint_set.limb_graph[line_index][0]],
                self.joint_positions[self.joint_set.limb_graph[line_index][1]]
            ], width=self.part_size / 2.2,
            color=int(skeleton_line_colors[line_index]),
            shader='mesh')
            for line_index in range(len(self.joint_set.limb_graph))
        ]

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
            shader='mesh')
            for line_index in range(len(self.joint_set.limb_graph))
        ]

    def __get_joint_names(self) -> List[Text]:
        return [k3d.text(
            text=self.joint_set.names[i],
            position=self.joint_positions[i],
            size=DEFAULT_TEXT_SIZE, label_box=False, color=DEFAULT_COLORS.get('black'))
            for i in range(self.joint_set.number_of_joints)
        ]

    def __get_joint_coordinates(self) -> List[Text]:
        return [k3d.text(
            text=COORDINATE_FORMAT.format(
                self.joint_positions[i][0],
                self.joint_positions[i][1],
                self.joint_positions[i][2]
            ), position= self.joint_positions[i],
            size=DEFAULT_TEXT_SIZE, label_box=False, color=DEFAULT_COLORS.get('black'))
            for i in range(self.joint_set.number_of_joints)
        ]

    def __get_joint_coordinates_for_video(self) -> List[Text]:
        return [k3d.text(
            text={
                current_timestamp[0]: COORDINATE_FORMAT.format(
                    current_timestamp[1][i][0],
                    current_timestamp[1][i][0],
                    current_timestamp[1][i][0])
                for current_timestamp in self.joint_positions.items()
            }, position={
                current_timestamp[0]: current_timestamp[1][i]
                for current_timestamp in self.joint_positions.items()
            }, size=DEFAULT_TEXT_SIZE, label_box=False, color=DEFAULT_COLORS.get('black'))
            for i in range(self.joint_set.number_of_joints)
        ]

    def __get_joint_colors(self) -> np.ndarray:
        joint_colors_shape: Tuple[int] = (self.joint_set.number_of_joints,)
        if self.color == 'default':
            joint_colors = np.zeros(shape=joint_colors_shape, dtype='uint32')
            joint_colors[self.joint_set.left_joint_indices] = DEFAULT_COLORS.get('red')
            joint_colors[self.joint_set.right_joint_indices] = DEFAULT_COLORS.get('blue')
            joint_colors[self.joint_set.center_joint_indices] = DEFAULT_COLORS.get('white')
            return joint_colors
        else:
            return np.full(shape=joint_colors_shape,
                           fill_value=self.__get_color(self.color), dtype='uint32')

    def __get_line_colors(self) -> np.ndarray:
        line_colors_shape: Tuple[int] = (self.joint_set.number_of_joints - 1,)
        if self.color == 'default':
            line_colors = np.zeros(shape=line_colors_shape, dtype='uint32')
            line_colors[self.joint_set.left_line_indices] = DEFAULT_COLORS.get('red')
            line_colors[self.joint_set.right_line_indices] = DEFAULT_COLORS.get('blue')
            line_colors[self.joint_set.center_line_indices] = DEFAULT_COLORS.get('white')
            return line_colors
        else:
            return np.full(
                shape=line_colors_shape,
                fill_value=self.__get_color(self.color),
                dtype='uint32'
            )

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
