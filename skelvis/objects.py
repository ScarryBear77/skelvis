from typing import Union, NewType, List, Dict, Final

import k3d
import numpy as np
from k3d.objects import Group, Line, Points, Text, Drawable

from .jointset import JointSet

Color = NewType('Color', Union[str, int])

DEFAULT_COLORS: Final[Dict[str, int]] = {
    'red': 0xFF0000, 'green': 0x00FF00,  'blue': 0x0000FF,
    'yellow': 0xFFFF00, 'teal': 0x00FFFF, 'purple': 0xFF00FF,
    'white': 0xFFFFFF, 'black': 0x000000
}


class DrawableSkeleton(Group):
    def __init__(self, joint_points: Points, joint_lines: List[Line], joint_names: List[Text] = None):
        points: List[Drawable] = [joint_points]
        drawable_objects: List[Drawable] = points + joint_lines
        if joint_names is not None:
            drawable_objects += joint_names
        super().__init__(drawable_objects)
        self.joint_points: Points = joint_points
        self.joint_lines: List[Line] = joint_lines
        self.joint_names: List[Text] = joint_names


class Skeleton:
    def __init__(self, joint_coordinates: np.ndarray,
                 joint_set: JointSet, part_size: float, color: Color):
        self.joint_coordinates: np.ndarray = joint_coordinates
        self.joint_set: JointSet = joint_set
        self.part_size: float = part_size
        self.color: Color = color

    def to_drawable_skeleton(self) -> DrawableSkeleton:
        joint_points = self.get_joint_points()
        joint_lines: List[Line] = self.get_joint_lines()
        return DrawableSkeleton(joint_points, joint_lines)

    def to_drawable_skeleton_with_names(self) -> DrawableSkeleton:
        joint_points = self.get_joint_points()
        joint_lines: List[Line] = self.get_joint_lines()
        joint_names: List[Text] = self.get_joint_names()
        return DrawableSkeleton(joint_points, joint_lines, joint_names)

    def get_joint_colors(self) -> np.ndarray:
        joint_colors_shape = (self.joint_set.number_of_joints, )
        if self.color == 'default':
            joint_colors = np.zeros(shape=joint_colors_shape, dtype='uint32')
            joint_colors[self.joint_set.left_joint_indices] = 0xFF0000
            joint_colors[self.joint_set.right_joint_indices] = 0x0000FF
            joint_colors[self.joint_set.center_joint_indices] = 0xFFFFFF
            return joint_colors
        else:
            return np.full(shape=joint_colors_shape,
                           fill_value=self.__get_color(self.color), dtype='uint32')

    def get_line_colors(self) -> np.ndarray:
        line_colors_shape = (self.joint_set.number_of_joints - 1, )
        if self.color == 'default':
            line_colors = np.zeros(shape=line_colors_shape, dtype='uint32')
            line_colors[self.joint_set.left_line_indices] = 0xFF0000
            line_colors[self.joint_set.right_line_indices] = 0x0000FF
            line_colors[self.joint_set.center_line_indices] = 0xFFFFFF
            return line_colors
        else:
            return np.full(shape=line_colors_shape,
                           fill_value=self.__get_color(self.color), dtype='uint32')

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

    def get_joint_points(self) -> Points:
        skeleton_joint_colors = self.get_joint_colors()
        return k3d.points(
            positions=self.joint_coordinates, point_size=self.part_size,
            shader='3d', colors=skeleton_joint_colors
        )

    def get_joint_lines(self) -> List[Line]:
        skeleton_line_colors = self.get_line_colors()
        return [self.__create_line_between_joints(
            start=self.joint_coordinates[self.joint_set.limb_graph[i][0]],
            end=self.joint_coordinates[self.joint_set.limb_graph[i][1]],
            width=self.part_size / 2.2,
            color=int(skeleton_line_colors[i]))
            for i in range(len(self.joint_set.limb_graph))
        ]

    def get_joint_names(self) -> List[Text]:
        return [k3d.text(
            text=self.joint_set.names[i],
            position=self.joint_coordinates[i],
            size=0.45, label_box=False, color=DEFAULT_COLORS.get('black'))
            for i in range(self.joint_set.number_of_joints)
        ]

    @staticmethod
    def __create_line_between_joints(start: float, end: float, width: float, color: int) -> Line:
        return k3d.line(
            vertices=[start, end], shader='thick',
            width=width, color=color
        )
