import numpy as np
import k3d
from k3d import objects
from abc import ABCMeta, abstractmethod


def get_skeleton_color(skeleton):
    if skeleton.color == 'default':
        return 0x0000FF
    else:
        return skeleton.color


def create_skeleton_joint_points(skeleton):
    return k3d.points(
            positions=skeleton.joint_coordinates, point_size=skeleton.part_size,
            shader='mesh', color=get_skeleton_color(skeleton)
        )


def create_skeleton_lines(skeleton, limb_graph):
    return [line for line in map(
            lambda line_indices: create_line_between_joints(
                start=skeleton.joint_coordinates[line_indices[0]],
                end=skeleton.joint_coordinates[line_indices[1]],
                width=skeleton.part_size / 4.0, color=get_skeleton_color(skeleton)
            ),
            limb_graph)]


def create_line_between_joints(start, end, width, color):
    return k3d.line(
        vertices=[start, end], shader='mesh',
        width=width, color=color
    )


class AbstractJointSet(metaclass=ABCMeta):
    def __init__(self):
        self.limb_graph = None
        self.number_of_joints = None

    @abstractmethod
    def convert_to_common_14(self):
        return None

    def generate_skeleton_plot(self, skeleton):
        assert skeleton.joint_coordinates.shape == (self.number_of_joints, 3)
        joint_points = create_skeleton_joint_points(skeleton)
        lines_between_joint_points = create_skeleton_lines(skeleton, self.limb_graph)
        skeleton_plot = objects.Group()
        skeleton_plot += joint_points
        for line in lines_between_joint_points:
            skeleton_plot += line
        return skeleton_plot


class MuPoTSJoints(AbstractJointSet):
    def __init__(self):
        super().__init__()
        self.names = np.array([
            'HEAD TOP', 'NECK',                              # Head
            'RIGHT SHOULDER', 'RIGHT ELBOW', 'RIGHT WRIST',  # Right arm
            'LEFT SHOULDER', 'LEFT ELBOW', 'LEFT WRIST',     # Left arm
            'RIGHT HIP', 'RIGHT KNEE', 'RIGHT ANKLE',        # Right leg
            'LEFT HIP', 'LEFT KNEE', 'LEFT ANKLE',           # Left leg
            'HIP', 'SPINE', 'NOSE'                           # Spine
        ])
        self.number_of_joints = 17
        self.limb_graph = [
            (10, 9), (9, 8), (8, 14),       # Right leg
            (13, 12), (12, 11), (11, 14),   # Left leg
            (0, 16), (16, 1),               # Head to Thorax
            (1, 15), (15, 14),              # Thorax to Hip
            (4, 3), (3, 2), (2, 1),         # Right arm
            (7, 6), (6, 5), (5, 1),         # Left arm
        ]
        self.names.flags.writeable = False
        self.sidedness = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1]

    def convert_to_common_14(self):
        common14_index_order = [14, 8, 9, 10, 11, 12, 13, 1, 5, 6, 7, 2, 3, 4]
        return Common14Joints(names=self.names[common14_index_order])


class OpenPoseJoints(AbstractJointSet):
    def __init__(self):
        super().__init__()
        self.names = np.array([
            'NOSE', 'NECK',                                    # Head
            'RIGHT SHOULDER', 'RIGHT ELBOW', 'RIGHT WRIST',    # Right arm
            'LEFT SHOULDER', 'LEFT ELBOW', 'LEFT WRIST',       # Left arm
            'HIP',                                             # Hip
            'RIGHT HIP', 'RIGHT KNEE', 'RIGHT ANKLE',          # Right leg
            'LEFT HIP', 'LEFT KNEE', 'LEFT ANKLE',             # Left leg
            'RIGHT EYE', 'LEFT EYE', 'RIGHT EAR', 'LEFT EAR',  # Face
            'LEFT BIG TOE', 'LEFT SMALL TOE', 'LEFT HEEL',     # Left foot
            'RIGHT BIG TOE', 'RIGHT SMALL TOE', 'RIGHT HEEL'   # Right foot
        ])
        self.limb_graph = [
            (1, 0), (17, 15), (15, 0), (16, 0), (18, 16),               # Head
            (4, 3), (3, 2), (2, 1),                                     # Right arm
            (7, 6), (6, 5), (5, 1),                                     # Left arm
            (1, 8),                                                     # Spine
            (23, 22), (22, 11), (24, 11), (11, 10), (10, 9), (9, 8),    # Right leg
            (20, 19), (19, 14), (21, 14), (14, 13), (13, 12), (12, 8)   # Left leg
        ]
        self.number_of_joints = 25
        self.names.flags.writeable = False

    def convert_to_common_14(self):
        common14_index_order = [8, 9, 10, 11, 12, 13, 14, 1, 5, 6, 7, 2, 3, 4]
        return Common14Joints(names=self.names[common14_index_order])

    def convert_to_only_stable_joints(self):
        stable_joint_indices = np.arange(17)
        return self.names[stable_joint_indices]


class CocoExJoints(AbstractJointSet):
    def __init__(self):
        super().__init__()
        self.names = np.array([
            'NOSE',                             # Head
            'LEFT EYE', 'RIGHT EYE',            # Eyes
            'LEFT EAR', 'RIGHT EAR',            # Ears
            'LEFT SHOULDER', 'RIGHT SHOULDER',  # Shoulders
            'LEFT ELBOW', 'RIGHT ELBOW',        # Elbows
            'LEFT WRIST', 'RIGHT WRIST',        # Wrists
            'LEFT HIP', 'RIGHT HIP',            # Hip
            'LEFT KNEE', 'RIGHT KNEE',          # Knees
            'LEFT ANKLE', 'RIGHT ANKLE',        # Ankles
            'HIP', 'NECK'                       # Spine
        ])
        self.number_of_joints = 19
        self.limb_graph = [
            (0, 1), (1, 3),                 # Left face
            (0, 2), (2, 4),                 # Right face
            (0, 18), (18, 17),              # Spine
            (18, 5), (5, 7), (7, 9),        # Left arm
            (18, 6), (6, 8), (8, 10),       # Right arm
            (17, 11), (11, 13), (13, 15),   # Left leg
            (17, 12), (12, 14), (14, 16)    # Right leg
        ]
        self.names.flags.writeable = False
        self.sidedness = [1, 1, 0, 0, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]

    def convert_to_common_14(self):
        common14_index_order = [17, 12, 14, 16, 11, 13, 15, 18, 5, 7, 9, 6, 8, 10]
        return Common14Joints(names=self.names[common14_index_order])


class PanopticJoints(AbstractJointSet):
    def __init__(self):
        super().__init__()
        self.names = np.array([
            'NECK', 'NOSE', 'HIP',                           # Spine
            'LEFT SHOULDER', 'LEFT ELBOW', 'LEFT WRIST',     # Left arm
            'LEFT HIP', 'LEFT KNEE', 'LEFT ANKLE',           # Left leg
            'RIGHT SHOULDER', 'RIGHT ELBOW', 'RIGHT WRIST',  # Right arm
            'RIGHT HIP', 'RIGHT KNEE', 'RIGHT ANKLE',        # Right leg
            'LEFT EYE', 'LEFT EAR',                          # Left face
            'RIGHT EYE', 'RIGHT EAR'                         # Right face
        ])
        self.number_of_joints = 19
        self.limb_graph = [
            (0, 1), (0, 2),               # Spine
            (0, 3), (3, 4), (4, 5),       # Left arm
            (2, 6), (6, 7), (7, 8),       # Left leg
            (0, 9), (9, 10), (10, 11),    # Right arm
            (2, 12), (12, 13), (13, 14),  # Right leg
            (1, 15), (15, 16),            # Left face
            (1, 17), (17, 18)             # Right face
        ]
        self.names.flags.writeable = False
        self.sidedness = [2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

    def convert_to_common_14(self):
        common14_index_order = [2, 12, 13, 14, 6, 7, 8, 0, 3, 4, 5, 9, 10, 11]
        return Common14Joints(self.names[common14_index_order])


class Common14Joints(AbstractJointSet):
    def __init__(self, names=np.array([
        'HIP',                                          # Hip
        'RIGHT HIP', 'RIGHT KNEE', 'RIGHT ANKLE',       # Right leg
        'LEFT HIP', 'LEFT KNEE', 'LEFT ANKLE',          # Left leg
        'NECK',                                         # Head
        'LEFT SHOULDER', 'LEFT ELBOW', 'LEFT WRIST',    # Left arm
        'RIGHT SHOULDER', 'RIGHT ELBOW', 'RIGHT WRIST'  # Right arm
    ])):
        super().__init__()
        self.names = names
        self.number_of_joints = 14
        self.limb_graph = [
            (0, 1), (1, 2), (2, 3),      # Right leg
            (0, 4), (4, 5), (5, 6),      # Left leg
            (0, 7),                      # Spine
            (7, 8), (8, 9), (9, 10),     # Left arm
            (7, 11), (11, 12), (12, 13)  # Right arm
        ]
        self.names.flags.writeable = False
        self.sidedness = [0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0]

    def convert_to_common_14(self):
        return Common14Joints(names=self.names)
