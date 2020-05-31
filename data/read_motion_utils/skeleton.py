# -*- coding: utf-8 -*-
from .bone import *


min_limit = -360.0
max_limit = 360.0


class Skeleton(object):
    """
    def __init__(self, bone_num, root, bone_list, end_effector_list, bone_id_map):
        self.bone_num = bone_num
        assert isinstance(root, Bone)
        self.root = root
        assert isinstance(bone_list, list)
        self.bone_list = bone_list
        assert isinstance(end_effector_list, list)
        self.end_effector_list = end_effector_list
        assert isinstance(bone_id_map, dict)
        self.bone_id_map = bone_id_map
    """

    def __init__(self):
        self.bone_num = 0
        self.root = Bone()
        self.bone_list = []
        self.end_effector_list = []
        self.bone_id_map = {}

    def read_mskel(self, path):
        if path.strip().split(".")[-1] != "mskel":
            raise Exception("The file is not a skeleton file!")
        f = open(path, 'r')
        lines = f.readlines()
        bone_ind = 0
        hierarchy_ind = 0
        for line in lines:
            if line.strip() == "mskel":
                continue
            if line.strip()[0] == "#":
                continue
            split_line = line.strip().split(" ")
            if len(split_line) == 2 and split_line[0] == "joint":
                self.bone_num = int(split_line[1])
                continue
            if bone_ind < self.bone_num:
                if len(line) == 0 or line[0] == "#":
                    continue
                """
                bone = Bone(None, [], bone_ind, [int(split_line[14]), int(split_line[15]), int(split_line[16]),
                                         int(split_line[17]), int(split_line[18]), int(split_line[19])], split_line[0],
                            np.mat([[float(split_line[2]), float(split_line[3]), float(split_line[4])]]).T,
                            np.mat([[float(split_line[5]), float(split_line[6]), float(split_line[7])]]).T,
                            np.mat([[float(split_line[8]), float(split_line[9]), float(split_line[10])]]).T,
                            np.mat([[float(split_line[11]), float(split_line[12]), float(split_line[13])]]).T)
                """
                bone = Bone()
                bone.set_bone_id(bone_ind)
                bone.set_locked([int(split_line[14]), int(split_line[15]), int(split_line[16]),
                                 int(split_line[17]), int(split_line[18]), int(split_line[19])])
                bone.set_name(split_line[0])
                bone.set_translation(np.mat([[float(split_line[2]), float(split_line[3]), float(split_line[4])]]).T)
                bone.set_joint_rotate(np.mat([[float(split_line[5]), float(split_line[6]), float(split_line[7])]]).T)
                bone.set_rotate(np.mat([[float(split_line[8]), float(split_line[9]), float(split_line[10])]]).T)
                bone.set_axis_rotate(np.mat([[float(split_line[11]), float(split_line[12]), float(split_line[13])]]).T)
                bone.update_local_matrix()
                self.bone_list.append(bone)
                self.bone_id_map[split_line[0]] = bone_ind
                bone_ind += 1
            if len(split_line) == 1 and split_line[0] == "hierarchy":
                hierarchy_ind += 1
            if hierarchy_ind > 0 and len(split_line) == 2:
                bone_id0 = self.bone_id_map[split_line[0]]
                bone_id1 = self.bone_id_map[split_line[1]]
                self.bone_list[bone_id0].set_parent(self.bone_list[bone_id1])
                self.bone_list[bone_id0].set_parent_name(split_line[1])
                self.bone_list[bone_id1].add_child_name(split_line[0])
                self.bone_list[bone_id1].add_child(self.bone_list[bone_id0])

        for i in range(self.bone_num):
            if not self.bone_list[i].parent:
                self.root = self.bone_list[i]
            if len(self.bone_list[i].children) == 0:
                self.end_effector_list.append(self.bone_list[i])

    def set_pose_without_limit(self, frame):
        for bone_pose in frame:
            bone_ind = self.bone_id_map[bone_pose.bone_name]
            bone = self.bone_list[bone_ind]
            if not bone_pose.dof_locked[0] and not bone.dof_locked[0]:
                bone.trans[0] = bone_pose.tx
            if not bone_pose.dof_locked[1] and not bone.dof_locked[1]:
                bone.trans[1] = bone_pose.ty
            if not bone_pose.dof_locked[2] and not bone.dof_locked[2]:
                bone.trans[2] = bone_pose.tz
            if not bone_pose.dof_locked[3] and not bone.dof_locked[3]:
                # bone.rotate[0] = degree2radian(bone_pose.rx)
                bone.rotate[0] = bone_pose.rx
            if not bone_pose.dof_locked[4] and not bone.dof_locked[4]:
                # bone.rotate[1] = degree2radian(bone_pose.ry)
                bone.rotate[1] = bone_pose.ry
            if not bone_pose.dof_locked[5] and not bone.dof_locked[5]:
                # bone.rotate[2] = degree2radian(bone_pose.rz)
                bone.rotate[2] = bone_pose.rz
        self.update_pose_matrix()

    def dfs_update_pose_matrix(self, bone):
        bone.update_local_matrix()
        bone.update_world_matrix()
        for child_bone in bone.get_children():
            self.dfs_update_pose_matrix(child_bone)

    def update_pose_matrix(self):
        self.dfs_update_pose_matrix(self.root)

    def get_pose(self):
        frame = []
        for i in range(self.bone_num):
            bone_pose = BonePoseSPS()
            bone = self.bone_list[i]
            bone_pose.bone_name = bone.get_name()
            if not bone.get_locked(0):
                bone_pose.tx = bone.get_translation()[0, 0]
                bone_pose.dof_locked[0] = 0
            if not bone.get_locked(1):
                bone_pose.ty = bone.get_translation()[1, 0]
                bone_pose.dof_locked[1] = 0
            if not bone.get_locked(2):
                bone_pose.tz = bone.get_translation()[2, 0]
                bone_pose.dof_locked[2] = 0
            if not bone.get_locked(3):
                bone_pose.rx = bone.get_rotate()[0, 0]
                bone_pose.dof_locked[3] = 0
            if not bone.get_locked(4):
                bone_pose.ry = bone.get_rotate()[1, 0]
                bone_pose.dof_locked[4] = 0
            if not bone.get_locked(5):
                bone_pose.rz = bone.get_rotate()[2, 0]
                bone_pose.dof_locked[5] = 0
            frame.append(bone_pose)

        return frame

    def get_root(self):
        return self.root

    def get_bone_num(self):
        return self.bone_num

    def get_bone_by_name(self, name):
        t = self.bone_id_map.get(name)
        if t or t == 0:
            return self.bone_list[self.bone_id_map[name]]
        else:
            return None

    def get_bone_by_id(self, ind):
        if ind < 0 or ind >= self.bone_num:
            return None
        else:
            return self.bone_list[ind]

    def get_end_effector_list(self):
        return self.end_effector_list

    def get_bone_list(self):
        return self.bone_list
