from typing import Tuple, List

import numpy as np
from abc import ABCMeta, abstractmethod


class JointSet(metaclass=ABCMeta):
    def __init__(self):
        self.names: np.ndarray = np.array([])
        self.number_of_joints: int = 0
        self.limb_graph: List[Tuple[int, int]] = []
        self.left_joint_indices: List[int] = []
        self.right_joint_indices: List[int] = []
        self.center_joint_indices: List[int] = []
        self.left_line_indices: List[int] = []
        self.right_line_indices: List[int] = []
        self.center_line_indices: List[int] = []
        self.vertically_aligned_line_indices: List[Tuple[int, int]] = []

    @abstractmethod
    def convert_to_common_14(self):
        return self


class MuPoTSJoints(JointSet):
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
        self.names.flags.writeable = False
        self.limb_graph = [
            (10, 9), (9, 8), (8, 14),       # Right leg
            (13, 12), (12, 11), (11, 14),   # Left leg
            (0, 16), (16, 1),               # Head to Thorax
            (1, 15), (15, 14),              # Thorax to Hip
            (4, 3), (3, 2), (2, 1),         # Right arm
            (7, 6), (6, 5), (5, 1),         # Left arm
        ]
        self.number_of_joints = 17
        self.left_joint_indices = [5, 6, 7, 11, 12, 13]
        self.right_joint_indices = [2, 3, 4, 8, 9, 10]
        self.center_joint_indices = [0, 1, 14, 15, 16]
        self.left_line_indices = [3, 4, 5, 13, 14, 15]
        self.right_line_indices = [0, 1, 2, 10, 11, 12]
        self.center_line_indices = [6, 7, 8, 9]
        self.vertically_aligned_line_indices = [
            (16, 0), (1, 16), (15, 1), (14, 15),
            (13, 12), (12, 11), (10, 9), (9, 8)
        ]

    def convert_to_common_14(self):
        common14_index_order = [14, 8, 9, 10, 11, 12, 13, 1, 5, 6, 7, 2, 3, 4]
        return Common14Joints(names=self.names[common14_index_order])


class OpenPoseJoints(JointSet):
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
        self.names.flags.writeable = False
        self.limb_graph = [
            (1, 0), (17, 15), (15, 0), (16, 0), (18, 16),               # Head
            (4, 3), (3, 2), (2, 1),                                     # Right arm
            (7, 6), (6, 5), (5, 1),                                     # Left arm
            (1, 8),                                                     # Spine
            (23, 22), (22, 11), (24, 11), (11, 10), (10, 9), (9, 8),    # Right leg
            (20, 19), (19, 14), (21, 14), (14, 13), (13, 12), (12, 8)   # Left leg
        ]
        self.number_of_joints = 25
        self.left_joint_indices = [5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21]
        self.right_joint_indices = [2, 3, 4, 9, 10, 11, 15, 17, 22, 23, 34]
        self.center_joint_indices = [0, 1, 8]
        self.left_line_indices = [3, 4, 8, 9, 10, 18, 19, 20, 21, 22, 23]
        self.right_line_indices = [1, 2, 5, 6, 7, 12, 13, 14, 15, 16, 17]
        self.center_line_indices = [0, 11]
        self.vertically_aligned_line_indices = [
            (1, 0), (8, 1), (11, 10), (10, 9),
            (14, 13), (13, 12), (24, 11), (21, 14)
        ]

    def convert_to_common_14(self):
        common14_index_order = [8, 9, 10, 11, 12, 13, 14, 1, 5, 6, 7, 2, 3, 4]
        return Common14Joints(names=self.names[common14_index_order])

    def convert_to_only_stable_joints(self):
        stable_joint_indices = np.arange(17)
        return self.names[stable_joint_indices]


class CocoExJoints(JointSet):
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
        self.names.flags.writeable = False
        self.limb_graph = [
            (0, 1), (1, 3),                 # Left face
            (0, 2), (2, 4),                 # Right face
            (0, 18), (18, 17),              # Spine
            (18, 5), (5, 7), (7, 9),        # Left arm
            (18, 6), (6, 8), (8, 10),       # Right arm
            (17, 11), (11, 13), (13, 15),   # Left leg
            (17, 12), (12, 14), (14, 16)    # Right leg
        ]
        self.number_of_joints = 19
        self.left_joint_indices = [1, 3, 5, 7, 9, 11, 13, 15]
        self.right_joint_indices = [2, 4, 6, 8, 10, 12, 14, 16]
        self.center_joint_indices = [0, 17, 18]
        self.left_line_indices = [0, 1, 6, 7, 8, 12, 13, 14]
        self.right_line_indices = [2, 3, 9, 10, 11, 15, 16, 17]
        self.center_line_indices = [4, 5]
        self.vertically_aligned_line_indices = [
            (18, 0), (17, 18), (16, 14),
            (14, 12), (15, 13), (13, 11)
        ]

    def convert_to_common_14(self):
        common14_index_order = [17, 12, 14, 16, 11, 13, 15, 18, 5, 7, 9, 6, 8, 10]
        return Common14Joints(names=self.names[common14_index_order])


class PanopticJoints(JointSet):
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
        self.names.flags.writeable = False
        self.limb_graph = [
            (0, 1), (0, 2),               # Spine
            (0, 3), (3, 4), (4, 5),       # Left arm
            (2, 6), (6, 7), (7, 8),       # Left leg
            (0, 9), (9, 10), (10, 11),    # Right arm
            (2, 12), (12, 13), (13, 14),  # Right leg
            (1, 15), (15, 16),            # Left face
            (1, 17), (17, 18)             # Right face
        ]
        self.number_of_joints = 19
        self.left_joint_indices = [3, 4, 5, 6, 7, 8, 15, 16]
        self.right_joint_indices = [9, 10, 11, 12, 13, 14, 17, 18]
        self.center_joint_indices = [0, 1, 2]
        self.left_line_indices = [2, 3, 4, 5, 6, 7, 14, 15]
        self.right_line_indices = [8, 9, 10, 11, 12, 13, 16, 17]
        self.center_line_indices = [0, 1]
        self.vertically_aligned_line_indices = [
            (0, 1), (2, 0), (8, 7),
            (7, 6), (14, 13), (13, 12)
        ]

    def convert_to_common_14(self):
        common14_index_order = [2, 12, 13, 14, 6, 7, 8, 0, 3, 4, 5, 9, 10, 11]
        return Common14Joints(names=self.names[common14_index_order])


class Common14Joints(JointSet):
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
        self.names.flags.writeable = False
        self.limb_graph = [
            (0, 1), (1, 2), (2, 3),      # Right leg
            (0, 4), (4, 5), (5, 6),      # Left leg
            (0, 7),                      # Spine
            (7, 8), (8, 9), (9, 10),     # Left arm
            (7, 11), (11, 12), (12, 13)  # Right arm
        ]
        self.number_of_joints = 14
        self.left_joint_indices = [4, 5, 6, 8, 9, 10]
        self.right_joint_indices = [1, 2, 3, 11, 12, 13]
        self.center_joint_indices = [0, 7]
        self.left_line_indices = [3, 4, 5, 7, 8, 9]
        self.right_line_indices = [0, 1, 2, 10, 11, 12]
        self.center_line_indices = [6]
        self.vertically_aligned_line_indices = [(7, 0), (3, 2), (2, 1), (6, 5), (5, 4)]

    def convert_to_common_14(self):
        return Common14Joints(names=self.names)
