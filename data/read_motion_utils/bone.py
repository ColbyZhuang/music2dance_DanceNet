# -*- coding: utf-8 -*-
from .matrix import *


class Bone(object):
    def __init__(self):
        """
        :type parent: Bone
        children: list of Bone
        dof_locked: list of int, 0 means false and 1 means true
        name: str
        trans, joint_orient, rotate, axis_rotate: column vector of float with size 3
        axis_x_trans, axis_y_trans, axis_z_trans, axis_x_rotate, axis_y_rotate, axis_z_rotate: column vector
                                                                                               of float with size 3
        t_mat, jo_mat, ro_mat, r_mat, local_mat, world_mat: mat of float with size 4 * 4
        """
        self.parent = None
        self.parent_name = None
        self.children = []
        self.children_name = []
        self.bone_id = 0
        self.dof_locked = [1, 1, 1, 1, 1, 1]
        self.name = ""
        self.trans = np.mat([0.0, 0.0, 0.0]).T
        self.joint_rotate = np.mat([0.0, 0.0, 0.0]).T
        self.rotate = np.mat([0.0, 0.0, 0.0]).T
        self.axis_rotate = np.mat([0.0, 0.0, 0.0]).T
        # self.axis_x_trans = axis_x_trans
        # self.axis_y_trans = axis_y_trans
        # self.axis_z_trans = axis_z_trans
        # self.axis_x_rotate = axis_x_rotate
        # self.axis_y_rotate = axis_y_rotate
        # self.axis_z_rotate = axis_z_rotate
        self.t_mat = translation(self.trans[0, 0], self.trans[1, 0], self.trans[2, 0])
        self.jo_mat = rotation_from_angle(self.joint_rotate[0, 0], self.joint_rotate[1, 0], self.joint_rotate[2, 0])
        self.ro_mat = rotation_from_angle(self.axis_rotate[0, 0], self.axis_rotate[1, 0], self.axis_rotate[2, 0])
        self.r_mat = rotation_from_angle(self.rotate[0, 0], self.rotate[1, 0], self.rotate[2, 0])
        self.local_mat = self.t_mat * self.jo_mat * self.r_mat * self.ro_mat
        self.world_mat = self.local_mat
        self.sps_world_mat = self.world_mat

    def update_local_matrix(self):
        self.t_mat = translation(self.trans[0, 0], self.trans[1, 0], self.trans[2, 0])
        self.jo_mat = rotation_from_angle(self.joint_rotate[0, 0], self.joint_rotate[1, 0], self.joint_rotate[2, 0])
        self.r_mat = rotation_from_angle(self.rotate[0, 0], self.rotate[1, 0], self.rotate[2, 0])
        self.ro_mat = rotation_from_angle(self.axis_rotate[0, 0], self.axis_rotate[1, 0], self.axis_rotate[2, 0])
        self.local_mat = self.t_mat * self.jo_mat * self.r_mat * self.ro_mat

    def update_world_matrix(self):
        if self.parent:
            parent_world_mat = self.parent.world_mat
            self.world_mat = parent_world_mat * self.local_mat
        else:
            self.world_mat = self.local_mat
        arr_world_mat = self.world_mat.getA()
        for i in range(arr_world_mat.shape[0]):
            for j in range(arr_world_mat.shape[1]):
                if abs(arr_world_mat[i][j]) > 1e6 or abs(arr_world_mat[i][j]) < 1e-6:
                    arr_world_mat[i][j] = 0.
        self.world_mat = np.mat(arr_world_mat)


    def update_sps_world_matrix(self, rx, ry, rz):
        mat_rot = rotation_from_angle(rx, ry, rz)

        arr_local_mat = self.local_mat.getA()
        for i in range(arr_local_mat.shape[0]):
            for j in range(arr_local_mat.shape[1]):
                if abs(arr_local_mat[i][j]) > 1e7 or abs(arr_local_mat[i][j]) < 1e-7:
                    arr_local_mat[i][j] = 0.
        self.local_mat = np.mat(arr_local_mat)

        if self.parent:
            parent_sps_world_mat = self.parent.sps_world_mat
            self.sps_world_mat = parent_sps_world_mat * self.local_mat * mat_rot
        else:
            self.sps_world_mat = self.local_mat * mat_rot

    def set_parent(self, bone):
        self.parent = bone

    def set_parent_name(self, name):
        self.parent_name = name

    def add_child(self, bone):
        self.children.append(bone)

    def add_child_name(self, name):
        self.children_name.append(name)

    def set_bone_id(self, ind):
        self.bone_id = ind

    def get_bone_id(self):
        return self.bone_id

    def set_name(self, b_name):
        self.name = b_name

    def get_name(self):
        return self.name

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def set_parent_null(self):
        self.parent = None

    def get_translation(self):
        return self.trans

    def set_translation(self, new_trans):
        self.trans = new_trans

    def get_rotate(self):
        return self.rotate

    def set_rotate(self, new_rotate):
        if self.dof_locked[3] == 0:
            self.rotate[0, 0] = new_rotate[0]
        if self.dof_locked[4] == 0:
            self.rotate[1, 0] = new_rotate[1]
        if self.rotate[2, 0] == 0:
            self.rotate[2, 0] = new_rotate[2]

    def get_locked(self, ind):
        return self.dof_locked[ind]

    def set_locked(self, dof_locked):
        self.dof_locked = dof_locked

    def set_joint_rotate(self, j_ro):
        self.joint_rotate = j_ro

    def get_joint_rotate(self):
        return self.joint_rotate

    def set_axis_rotate(self, a_ro):
        self.axis_rotate = a_ro

    def get_axis_rotate(self):
        return self.axis_rotate

    def get_world_matrix(self):
        return self.world_mat

    def get_local_matrix(self):
        return self.local_mat

    def get_translation_matrix(self):
        return self.t_mat

    def get_rotate_matrix(self):
        return self.r_mat

    def get_ro_matrix(self):
        return self.ro_mat

    def get_jo_matrix(self):
        return self.jo_mat


