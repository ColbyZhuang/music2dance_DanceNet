from data.read_motion_utils.animation import *
from data.read_motion_utils.matrix import *
from scipy import linalg as la
import matplotlib.pyplot as plt
import math
import copy


def quat_add(sqs, oqs):
    q0 = sqs[..., 0];
    q1 = sqs[..., 1];
    q2 = sqs[..., 2];
    q3 = sqs[..., 3];
    r0 = oqs[..., 0];
    r1 = oqs[..., 1];
    r2 = oqs[..., 2];
    r3 = oqs[..., 3];
    qs = np.empty(sqs.shape)
    qs[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    qs[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    qs[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    qs[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
    q0_ = qs[..., 0].copy()
    qs[..., 0] = np.where(q0_ <= 0., -qs[..., 0], qs[..., 0])
    qs[..., 1] = np.where(q0_ <= 0., -qs[..., 1], qs[..., 1])
    qs[..., 2] = np.where(q0_ <= 0., -qs[..., 2], qs[..., 2])
    qs[..., 3] = np.where(q0_ <= 0., -qs[..., 3], qs[..., 3])
    return qs

def quat_sub(q1, q2):
    '''

    :param q1:
    :param q2:
    :return: q2 - q1
    '''
    q1_neg = q1 * np.array([[1, -1, -1, -1]])

    result_q = quat_add(q1_neg, q2)


    return result_q


def smooth_quat(qs):
    if qs[0, 0] < 0:
        qs[0] *= -1
    for i in range(1, qs.shape[0]):
        delta_q_1 = quat_sub(qs[i-1:i], qs[i:i+1])
        theta1 = np.arccos(delta_q_1[:, 0])

        delta_q_2 = quat_sub(qs[i - 1:i], -qs[i:i + 1])
        theta2 = np.arccos(delta_q_2[:, 0])

        if theta1 > theta2:
            qs[i] *= -1
    return qs



def mat2quaternion(ts):

    d0, d1, d2 = ts[..., 0, 0], ts[..., 1, 1], ts[..., 2, 2]

    q0 = (d0 + d1 + d2 + 1.0) / 4.0
    q1 = (d0 - d1 - d2 + 1.0) / 4.0
    q2 = (-d0 + d1 - d2 + 1.0) / 4.0
    q3 = (-d0 - d1 + d2 + 1.0) / 4.0

    q0 = np.sqrt(q0.clip(0, None))
    q1 = np.sqrt(q1.clip(0, None))
    q2 = np.sqrt(q2.clip(0, None))
    q3 = np.sqrt(q3.clip(0, None))

    c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3)
    c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
    c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3)
    c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)

    q1[c0] *= np.sign(ts[c0, 2, 1] - ts[c0, 1, 2])
    q2[c0] *= np.sign(ts[c0, 0, 2] - ts[c0, 2, 0])
    q3[c0] *= np.sign(ts[c0, 1, 0] - ts[c0, 0, 1])

    q0[c1] *= np.sign(ts[c1, 2, 1] - ts[c1, 1, 2])
    q2[c1] *= np.sign(ts[c1, 1, 0] + ts[c1, 0, 1])
    q3[c1] *= np.sign(ts[c1, 0, 2] + ts[c1, 2, 0])

    q0[c2] *= np.sign(ts[c2, 0, 2] - ts[c2, 2, 0])
    q1[c2] *= np.sign(ts[c2, 1, 0] + ts[c2, 0, 1])
    q3[c2] *= np.sign(ts[c2, 2, 1] + ts[c2, 1, 2])

    q0[c3] *= np.sign(ts[c3, 1, 0] - ts[c3, 0, 1])
    q1[c3] *= np.sign(ts[c3, 2, 0] + ts[c3, 0, 2])
    q2[c3] *= np.sign(ts[c3, 2, 1] + ts[c3, 1, 2])

    q0_ = q0.copy()
    q0 = np.where(q0_ <= 0., -q0, q0)
    q1 = np.where(q0_ <= 0., -q1, q1)
    q2 = np.where(q0_ <= 0., -q2, q2)
    q3 = np.where(q0_ <= 0., -q3, q3)


    qs = np.empty(ts.shape[:-2] + (4,))
    qs[..., 0] = q0
    qs[..., 1] = q1
    qs[..., 2] = q2
    qs[..., 3] = q3


    # normlize
    length = np.sum(qs ** 2.0, axis=-1) ** 0.5

    return qs / length[..., np.newaxis]





def quaternion2mat(qs):
    qw = qs[..., 0]
    qx = qs[..., 1]
    qy = qs[..., 2]
    qz = qs[..., 3]

    x2 = qx + qx;
    y2 = qy + qy;
    z2 = qz + qz;
    xx = qx * x2;
    yy = qy * y2;
    wx = qw * x2;
    xy = qx * y2;
    yz = qy * z2;
    wy = qw * y2;
    xz = qx * z2;
    zz = qz * z2;
    wz = qw * z2;

    m = np.empty(qs.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m

def quat2euler(qs):
    q0 = qs[..., 0]
    q1 = qs[..., 1]
    q2 = qs[..., 2]
    q3 = qs[..., 3]
    es = np.zeros(qs.shape[:-1] + (3,))
    es[..., 0] = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    es[..., 1] = np.arcsin((2 * (q0 * q2 - q3 * q1)).clip(-1, 1))
    es[..., 2] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    return es



def quaternion2RotVector(qs):
    '''
    function: log(qs)
    v = log( q ) = theta * v1, where theta = 2*arccos(q0), v1 = [q1 q2 q3]/sin(theta/2)
    :param qs:
    :return: RotVector
    '''
    theta = 2 * np.arccos(qs[..., 0])

    v1_0 = qs[..., 1] / np.sin(0.5 * theta)
    v1_1 = qs[..., 2] / np.sin(0.5 * theta)
    v1_2 = qs[..., 3] / np.sin(0.5 * theta)

    v1_0 = np.where(np.isnan(v1_0), 0., v1_0)
    v1_1 = np.where(np.isnan(v1_1), 0., v1_1)
    v1_2 = np.where(np.isnan(v1_2), 0., v1_2)


    v = np.empty(qs.shape[:-1] + (3,))
    v[..., 0] = theta * v1_0
    v[..., 1] = theta * v1_1
    v[..., 2] = theta * v1_2

    v[..., 0] = np.where(np.isnan(v[..., 0]), 0., v[..., 0])
    v[..., 1] = np.where(np.isnan(v[..., 1]), 0., v[..., 1])
    v[..., 2] = np.where(np.isnan(v[..., 2]), 0., v[..., 2])
    print(np.where(np.isnan(v)))

    return v

def RotVector2quaternion(v):
    '''
    q = exp( v ) = [ sin(theta/2)*v/theta cos(theta/2) ], where theta = |v|
    :param v: rotation vector
    :return: exp( v )
    '''
    theta = np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2])
    q0 = np.cos(0.5 * theta)
    q1 = v[..., 0] * np.sin(0.5 * theta) / theta
    q2 = v[..., 1] * np.sin(0.5 * theta) / theta
    q3 = v[..., 2] * np.sin(0.5 * theta) / theta

    # q0 = np.where(theta <= 0.00001, 1., q0)
    # q1 = np.where(theta <= 0.00001, 0., q1)
    # q2 = np.where(theta <= 0.00001, 0., q2)
    # q3 = np.where(theta <= 0.00001, 0., q3)

    q0 = np.where(np.isnan(q0), 1., q0)
    q1 = np.where(np.isnan(q1), 0., q1)
    q2 = np.where(np.isnan(q2), 0., q2)
    q3 = np.where(np.isnan(q3), 0., q3)



    qs = np.empty(v.shape[:-1] + (4,))
    qs[..., 0] = q0
    qs[..., 1] = q1
    qs[..., 2] = q2
    qs[..., 3] = q3

    # normlize
    length = np.sum(qs ** 2.0, axis=-1) ** 0.5
    qs_norm = qs / length[..., np.newaxis]
    return qs_norm


def getqmean(qs):
    _, _, v = la.svd(qs, 0)
    qmean = v[0]
    return qmean


class ani_process(object):
    def __init__(self):
        pass
    def read_poses(self, sps_path):
        get_motion = SPSPose()
        get_motion.read_sps_pose(sps_path)
        poses = get_motion.poses
        poses_r_mat = []
        root_tran = []
        root_rx = []
        root_ry = []
        for pose in poses:
            root_tran.append([pose[0].tx, pose[0].ty, pose[0].tz])
            root_rx.append(rotation_from_angle(0, pose[0].rx, 0))
            root_ry.append(rotation_from_angle(pose[0].rx, 0, 0))
            pose_r_mat = []
            for p in pose:
                r_mat = rotation_from_angle(p.rx, p.ry, p.rz)
                pose_r_mat.append(r_mat)
            poses_r_mat.append(pose_r_mat)
        poses_r_arr = np.asarray(poses_r_mat, dtype=np.float32)
        root_tran_arr = np.asarray(root_tran, dtype=np.float32)
        root_rx_arr = np.asarray(root_rx, dtype=np.float32)
        root_ry_arr = np.asarray(root_ry, dtype=np.float32)
        return poses_r_arr, root_tran_arr, root_rx_arr, root_ry_arr

    def root_tran_deform(self, roots_tran, root_rx_arr):
        face_angle_root_tran = np.zeros((roots_tran.shape[0], 4))
        root_rx_quat = self.poseMat2quat(root_rx_arr)
        root_rx_quat = smooth_quat(root_rx_quat)
        delta_root_rx_quat = quat_sub(root_rx_quat[:-1], root_rx_quat[1:])
        delta_root_rx_vector = self.quat2rotVector(delta_root_rx_quat)

        root_rx_vector = self.quat2rotVector(root_rx_quat)
        roots_tran[1:, 0] -= copy.deepcopy(roots_tran[:-1, 0])
        roots_tran[1:, 2] -= copy.deepcopy(roots_tran[:-1, 2])
        roots_tran[:, 0] = roots_tran[:, 0] / 10.
        roots_tran[:, 1] = roots_tran[:, 1] / 1000.
        roots_tran[:, 2] = roots_tran[:, 2] / 10.

        for i in range(roots_tran.shape[0]):
            if i == 0:
                face_angle_root_tran[0, 0] = root_rx_vector[0, 1]
                face_angle_root_tran[0, 1] = roots_tran[0, 0]
                face_angle_root_tran[0, 2] = roots_tran[0, 1]
                face_angle_root_tran[0, 3] = roots_tran[0, 2]
            else:
                face_angle_root_tran[i, 0] = delta_root_rx_vector[i-1, 1]
                root_tran_vec = (np.mat(root_rx_arr[i-1]).I) * np.mat([roots_tran[i, 0], 0.0, roots_tran[i, 2], 1.0]).T

                face_angle_root_tran[i, 1] = root_tran_vec[0]
                face_angle_root_tran[i, 2] = roots_tran[i, 1]
                face_angle_root_tran[i, 3] = root_tran_vec[2]

        return face_angle_root_tran

    def root_tran_recovery(self, face_angle_root_tran):
        root_tran = np.zeros((face_angle_root_tran.shape[0], 3))
        rot_vector = np.zeros((face_angle_root_tran.shape[0], 3))
        rot_vector[:, 1] = face_angle_root_tran[:, 0]
        rot_quat = self.rotVector2quat(rot_vector)
        for i in range(1, rot_quat.shape[0]):
            rot_quat[i] = quat_add(rot_quat[i-1], rot_quat[i])
            rot_mat = self.quat2poseRarr(rot_quat[i])
            root_tran[i] = (np.asarray((np.mat(rot_mat)) * np.mat([face_angle_root_tran[i, 1], 0.0, face_angle_root_tran[i, 3]]).T, dtype=np.float32))[:, 0]

        root_tran[:, 0] *= 10.
        root_tran[:, 1] = 1000. * face_angle_root_tran[:, 2]
        root_tran[:, 2] *= 10.
        for i in range(root_tran.shape[0] - 1):
            root_tran[i+1, 0] += root_tran[i, 0]
            root_tran[i + 1, 2] += root_tran[i, 2]

        return root_tran

    def root_rot_deform(self, root_rot, root_ry_arr, root_qmean):
        root_rot_quat = self.poseMat2quat(root_rot)
        root_rot_quat = smooth_quat(root_rot_quat)
        root_ry_quat = self.poseMat2quat(root_ry_arr)
        root_ry_quat = smooth_quat(root_ry_quat)
        quat_sub_y = quat_sub(root_ry_quat, root_rot_quat)
        quat_sub_y_sub_mean = quat_sub(np.tile(root_qmean, (quat_sub_y.shape[0], 1)), quat_sub_y)
        root_rot_vector = self.quat2rotVector(quat_sub_y_sub_mean)
        return root_rot_vector

    def root_rot_recovery(self, root_rot_vector, face_angle_root_tran, root_qmean):
        temp_face_angle_vec = np.zeros([face_angle_root_tran.shape[0], 3])
        temp_face_angle_vec[:, 0] = face_angle_root_tran[:, 0]
        face_angle_quat = self.rotVector2quat(temp_face_angle_vec)
        for i in range(1, face_angle_quat.shape[0]):
            face_angle_quat[i] = quat_add(face_angle_quat[i - 1], face_angle_quat[i])
        root_rot_quat = self.rotVector2quat(root_rot_vector)
        root_rot_quat = quat_add(np.tile(root_qmean, (root_rot_quat.shape[0], 1)), root_rot_quat)
        root_rot_quat = quat_add(face_angle_quat, root_rot_quat)
        root_rot_mat_arr = self.quat2poseRarr(root_rot_quat)
        return root_rot_mat_arr

    def get_meanq(self, poses_r_arr, root_ry_arr):
        root_rot = poses_r_arr[:,0]
        root_rot_quat = self.poseMat2quat(root_rot)
        root_rot_quat = smooth_quat(root_rot_quat)
        root_ry_quat = self.poseMat2quat(root_ry_arr)
        root_ry_quat = smooth_quat(root_ry_quat)
        root_qsub = quat_sub(root_ry_quat, root_rot_quat)
        root_qmean = getqmean(root_qsub)

        joint_qmean = np.zeros((poses_r_arr[:,1:].shape[1], 4))
        joint_rot_quat = self.poseMat2quat(poses_r_arr[:,1:])
        for j in range(joint_rot_quat.shape[1]):
            joint_rot_quat[:, j] = smooth_quat(joint_rot_quat[:, j])
            joint_qmean[j] = getqmean(joint_rot_quat[:, j])

        root_qmean = np.expand_dims(root_qmean, axis=0)
        all_qmean = np.concatenate((root_qmean, joint_qmean), axis=0)

        return all_qmean

    def get_all_meanq(self, poses_r_arr, root_ry_arr):
        root_qsub_all = []
        joint_q_all = []
        for i in range(poses_r_arr.shape[0]):
            root_rot = poses_r_arr[i][:,0]
            root_rot_quat = self.poseMat2quat(root_rot)
            root_rot_quat = smooth_quat(root_rot_quat)
            root_ry_quat = self.poseMat2quat(root_ry_arr[i])
            root_ry_quat = smooth_quat(root_ry_quat)
            root_qsub = quat_sub(root_ry_quat, root_rot_quat)

            joint_rot_quat = self.poseMat2quat(poses_r_arr[i][:, 1:])
            for j in range(joint_rot_quat.shape[1]):
                joint_rot_quat[:, j] = smooth_quat(joint_rot_quat[:, j])

            if i == 0:
                root_qsub_all = root_qsub
                joint_q_all = joint_rot_quat
            else:
                root_qsub_all = np.concatenate((root_qsub_all, root_qsub), axis=0)
                joint_q_all = np.concatenate((joint_q_all, joint_rot_quat), axis=0)




        root_qmean = getqmean(root_qsub_all)

        joint_qmean = np.zeros((poses_r_arr[0][:,1:].shape[1], 4))
        for j in range(joint_q_all.shape[1]):
            joint_qmean[j] = getqmean(joint_q_all[:, j])

        root_qmean = np.expand_dims(root_qmean, axis=0)
        all_qmean = np.concatenate((root_qmean, joint_qmean), axis=0)

        return all_qmean

    def get_foot_contact(self, path):
        foot_cat = get_motion_bone_position(path)
        return foot_cat.foot_contact()


    def joint_mat2vector(self, joint_rot_mat_arr, joint_rot_qmean):
        joint_rot_quat = self.poseMat2quat(joint_rot_mat_arr)
        for j in range(joint_rot_quat.shape[1]):
            joint_rot_quat[:, j] = smooth_quat(joint_rot_quat[:, j])
            joint_rot_quat[:, j] = quat_sub(np.tile(joint_rot_qmean[j], (joint_rot_quat.shape[0], 1)), joint_rot_quat[:, j])
            joint_rot_quat[:, j] = smooth_quat(joint_rot_quat[:, j])

        joint_rot_vector = self.quat2rotVector(joint_rot_quat)
        return joint_rot_vector

    def joint_vector2mat(self, joint_rot_vector, joint_rot_qmean):
        joint_rot_quat = self.rotVector2quat(joint_rot_vector)
        joint_rot_qmean_ = np.tile(joint_rot_qmean, (joint_rot_quat.shape[0], 1, 1))
        joint_rot_quat = quat_add(joint_rot_qmean_, joint_rot_quat)
        joint_rot_mat_arr = self.quat2poseRarr(joint_rot_quat)
        return joint_rot_mat_arr

    def poseMat2quat(self, poses_r_arr):
        poses_quat = mat2quaternion(poses_r_arr)
        return poses_quat

    def quat2poseRarr(self, poses_quat):
        poses_r_array = quaternion2mat(poses_quat)
        return poses_r_array


    def quat2rotVector(self, poses_quat):
        rotVector = quaternion2RotVector(poses_quat)
        return rotVector

    def rotVector2quat(self, rotVector):
        poses_quat = RotVector2quaternion(rotVector)
        return poses_quat

    def rot_first_frame_to_faceZ(self, poses_r_arr, root_tran_arr, path):
        import copy
        root_rot_arr = poses_r_arr[:, 0]
        root_rot_f0 = copy.deepcopy(root_rot_arr[0])
        angle = rotation2angle(root_rot_arr[0])
        rot_mat = rotation_from_angle(0, -angle[0], 0)
        for i in range(root_rot_arr.shape[0]):
            root_rot_arr[i] = np.mat(root_rot_f0).I * root_rot_arr[i]

            deform_mat2_arr = np.array(rot_mat[:3, :3] * np.mat(root_tran_arr[i]).T)

            root_tran_arr[i] = deform_mat2_arr[:, 0]

        root_tran_arr[:, 0] = root_tran_arr[:, 0] - np.tile(root_tran_arr[0, 0], (root_tran_arr.shape[0]))
        root_tran_arr[:, 2] = root_tran_arr[:, 2] - np.tile(root_tran_arr[0, 2], (root_tran_arr.shape[0]))
        write2sps(root_tran_arr, poses_r_arr, path)




def write2sps(root_tran, poses_r_arry, path):
    import os
    if os.path.exists(path):
        os.remove(path)
    from data.read_motion_utils.animation import SPSPose
    from data.read_motion_utils import matrix
    import copy
    get_poses = SPSPose()
    get_poses.read_sps_pose("./data/standrand_sps.sps")
    std_pose_num = get_poses.pose_num
    std_poses = get_poses.poses
    std_key_vec = get_poses.key_vec

    write_poses = SPSPose()
    write_poses.pose_num = root_tran.shape[0]
    write_poses.key_vec = (np.arange(1, root_tran.shape[0]+1)).tolist()
    write_poses_poses = []
    for i in range(write_poses.pose_num):
        std_pose = copy.deepcopy(std_poses[0])
        for j, p in enumerate(std_pose):
            if p.bone_name == "root":
                p.tx = root_tran[i, 0]
                p.ty = root_tran[i, 1]
                p.tz = root_tran[i, 2]
            r_mat = np.asmatrix(poses_r_arry[i, j])
            euler = matrix.rotation2angle(r_mat)
            p.rx = euler[0]
            p.ry = euler[1]
            p.rz = euler[2]
        write_poses_poses.append(std_pose)
    write_poses.poses = copy.deepcopy(write_poses_poses)
    write_poses.write_sps_pose(path)

    print("22")


if __name__ == '__main__':
    import copy
    ani_pro = ani_process()
    poses_r_arr, root_tran_arr, root_rx, root_ry = ani_pro.read_poses("./data/walk/walk1_faceZ_clip.sps")
    #ani_pro.rot_first_frame_to_faceZ(poses_r_arr, root_tran_arr, path = "./data/test_walk_gt_faceZ.sps")

    all_quat_mean = ani_pro.get_meanq(poses_r_arr, root_ry)
    face_angel_root_deform = ani_pro.root_tran_deform(root_tran_arr, root_rx)

    root_rot_deform = ani_pro.root_rot_deform(poses_r_arr[:, 0], root_ry, all_quat_mean[0])

    joint_rot_deform = ani_pro.joint_mat2vector(poses_r_arr[:,1:], all_quat_mean[1:])
    joint_rot_deform_reshape = joint_rot_deform.reshape(
        (joint_rot_deform.shape[0], joint_rot_deform.shape[1] * 3))

    motion_representation_ = np.concatenate((face_angel_root_deform, root_rot_deform, joint_rot_deform_reshape), axis=1)


    motion_representation = motion_representation_[:]
    path = "./data/walk/walk1_faceZ_clip_sub_qmean/"
    for i in range(motion_representation.shape[1]):
        plt.clf()
        plt.plot(motion_representation[:, i])
        plt.savefig(path + str(i) + ".png")

    root_trans_recovery = ani_pro.root_tran_recovery(motion_representation[:, :4])
    root_rot_recovery = ani_pro.root_rot_recovery(motion_representation[:, 4:7], motion_representation[:, :4], all_quat_mean[0])

    motion_representation_joint = motion_representation[:, 7:].reshape((motion_representation.shape[0], int((motion_representation[:, 7:].shape[1])/3), 3))

    joint_rot_recovery = ani_pro.joint_vector2mat(motion_representation_joint, all_quat_mean[1:])

    root_rot_recovery = np.expand_dims(root_rot_recovery, axis=1)
    rot_recovery = np.concatenate((root_rot_recovery, joint_rot_recovery), axis=1)


    path = "./data/walk/walk1faceZ_clip_recovery.sps"
    write2sps(root_trans_recovery, rot_recovery, path)
    print("222")




