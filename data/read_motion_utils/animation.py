# -*- coding: utf-8 -*-
from data.read_motion_utils.skeleton import Skeleton
import numpy as np



class BonePoseSPS(object):
    def __init__(self):
        self.bone_name = ""
        self.dof_locked = [1, 1, 1, 1, 1, 1]
        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.rz = 0.0

        self.world_mat = np.mat([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])

    def set_name(self, name):
        self.bone_name = name

    def set_dof(self, dof):
        self.dof_locked = dof



def is_number(s):
    flag = True
    for i in range(len(s)):
        if not ("0" <= s[i] <= "9"):
            flag = False
            break
    return flag


class SPSPose(object):
    def __init__(self):
        self.pose_num = 0
        self.poses = []
        self.key_vec = []

    def read_sps_pose(self, path):
        if path.strip().split(".")[-1] != "sps":
            raise Exception("The file is not a pose file!")
        f = open(path, 'r')
        lines = f.readlines()
        empty_num = 0
        pose_begin = False
        first_begin = False
        last_frame_num = 0
        frame = []
        for line in lines:
            if line.strip()[0] == "#":
                continue
            split_line = line.strip().split(" ")
            if len(split_line) == 1 and not is_number(split_line[0]):
                continue
            if len(split_line) == 1 and is_number(split_line[0]):
                cur_frame_num = int(split_line[0])
                if cur_frame_num > last_frame_num + 1:
                    for i in range(cur_frame_num - last_frame_num - 1):
                        self.poses.append([])

                last_frame_num = cur_frame_num
                if pose_begin and not first_begin:
                    empty_num += 1
                elif pose_begin:
                    self.poses.append([])
                pose_begin = True
                if cur_frame_num != 1:
                    self.poses.append(frame)
                    self.key_vec.append(cur_frame_num)
                frame = []
                continue
            if pose_begin:
                # pose_begin = False
                bone_pose = BonePoseSPS()
                bone_pose.bone_name = split_line[0]
                for i in range(len(split_line)):
                    split_value = split_line[i].split(":")
                    if split_value[0] == "tx":
                        bone_pose.dof_locked[0] = 0
                        bone_pose.tx = float(split_value[1])
                    elif split_value[0] == "ty":
                        bone_pose.dof_locked[1] = 0
                        bone_pose.ty = float(split_value[1])
                    elif split_value[0] == "tz":
                        bone_pose.dof_locked[2] = 0
                        bone_pose.tz = float(split_value[1])
                    elif split_value[0] == "rx":
                        bone_pose.dof_locked[3] = 0
                        bone_pose.rx = float(split_value[1])
                    elif split_value[0] == "ry":
                        bone_pose.dof_locked[4] = 0
                        bone_pose.ry = float(split_value[1])
                    elif split_value[0] == "rz":
                        bone_pose.dof_locked[5] = 0
                        bone_pose.rz = float(split_value[1])
                frame.append(bone_pose)
        self.poses.append(frame)
        self.pose_num = len(self.poses)



    def write_sps_pose(self, path):
        f = open(path, 'a')
        f.write("skel_poseset_v01\n")
        #last_ind = 0
        for i in range(len(self.key_vec)):
            ind = self.key_vec[i] - 1
            # for j in range(last_ind, ind):
            #     f.write(str(j) + "\n")
            #last_ind = ind + 1
            #bone_pose = self.poses[ind]
            #f.write(str(ind + 1) + "\n")
            bone_pose = self.poses[i]
            f.write(str(i + 1) + "\n")

            for j in range(len(bone_pose)):
                no_degree = True
                for k in range(6):
                    if not bone_pose[j].dof_locked[k]:
                        no_degree = False
                        break
                if no_degree:
                    continue
                f.write(bone_pose[j].bone_name + " ")
                if not bone_pose[j].dof_locked[0]:
                    f.write("tx:" + str(bone_pose[j].tx) + " ")
                if not bone_pose[j].dof_locked[1]:
                    f.write("ty:" + str(bone_pose[j].ty) + " ")
                if not bone_pose[j].dof_locked[2]:
                    f.write("tz:" + str(bone_pose[j].tz) + " ")
                if not bone_pose[j].dof_locked[3]:
                    f.write("rx:" + str(bone_pose[j].rx) + " ")
                if not bone_pose[j].dof_locked[4]:
                    f.write("ry:" + str(bone_pose[j].ry) + " ")
                if not bone_pose[j].dof_locked[5]:
                    f.write("rz:" + str(bone_pose[j].rz) + " ")
                f.write("\n")
        f.close()



    def get_pose_num(self):
        return self.pose_num

    def get_frame(self, pose_id):
        return self.poses[pose_id]

    def get_poses(self):
        return self.poses

    def get_key_vec(self):
        return self.key_vec

    def add_frame(self, frame):
        self.poses.append(frame)
        self.pose_num += 1

    def add_frame_key(self, key):
        self.key_vec.append(key)

class get_motion_bone_position(object):
    def __init__(self, path, mskel_name):
        self.sps_path = path
        self.get_skeleton = Skeleton()
        self.get_skeleton.read_mskel("./data/" + mskel_name + ".mskel")
        self.bone_num = self.get_skeleton.bone_num
        self.root = self.get_skeleton.root
        self.bone_list = self.get_skeleton.bone_list

        self.get_motion = SPSPose()
        self.get_motion.read_sps_pose(self.sps_path)
        self.pose_num = self.get_motion.pose_num
        self.poses = self.get_motion.poses
        self.key_vec = self.get_motion.key_vec

        self.frame = 0



    def one_frame_bone_position(self, pose):
        bones_position = []
        for bone in self.bone_list:
            b_flag = False
            for p in pose:
                if bone.name == p.bone_name:
                    bone.update_sps_world_matrix(p.rx, p.ry, p.rz)
                    b_flag = True
                    break
            if b_flag == False:
                bone.update_sps_world_matrix(0, 0, 0)

            bone_position = bone.sps_world_mat[:3, 3].tolist()
            bones_position.append([bone_position[0][0] + pose[0].tx, bone_position[1][0] + pose[0].ty, bone_position[2][0] + pose[0].tz])
        return bones_position



    def get_bone_id(self, bone):
        return bone.bone_id

    def get_bone_pair(self, bones_position):
        bones_pair = []
        for bone in self.bone_list:
            if bone.children:
                for child in bone.children:
                    bones_pair.append([bones_position[self.get_bone_id(bone)], bones_position[self.get_bone_id(child)]])
        return bones_pair

    def get_sps_motion(self, frame):
        pose = self.poses[frame]
        one_frame_bone_position = self.one_frame_bone_position(pose)
        one_frame_bone_pair = self.get_bone_pair(one_frame_bone_position)
        return one_frame_bone_position, one_frame_bone_pair





