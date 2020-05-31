# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import torch
import torch.utils.data as data
from glob import glob
from scipy.interpolate import interp1d
import madmom

from utils.osutils import *
from data.read_motion_utils.animation_process import *
from utils.hparams import Hparams


class dance_motion_music_pre(data.Dataset):
    def __init__(self, train=True):
        hparams = Hparams()
        self.hparams = hparams
        self.is_train = train
        self.moving_window = 3
        self.moving_window_chroma = 0.5
        self.sequence_length = 480
        self.sequence_chroma_length = 80
        self.ani_process = ani_process()
        self.path = hparams.data_path
        #self.path = './motion_music_align_fix/motion_align_and_mirror/'
        train_path = glob(os.path.join(self.path, '*.sps'))

        music_wave_path = hparams.music_wave_path
        moving_weight_mean = np.load(self.path + "480_30_distance_60kmean_mean_p.npy")
        # self.motion_name = ["dance2_1_01_faceZ", "dance2_1_02_faceZ", "dance2_2_01_faceZ", "dance2_2_02_faceZ", "dance2_3_01_faceZ", "dance2_3_02_faceZ", "dance2_3_03_faceZ", "dance2_4_01_faceZ", "dance2_4_02_faceZ", "dance2_4_03_faceZ",\
        #                     "dance2_5_01_faceZ", "dance2_5_02_faceZ", "dance2_6_01_faceZ", "dance2_6_02_faceZ", "dance2_7_01_faceZ", "dance2_7_02_faceZ", "dance2_8_01_faceZ"]
        # self.music_name = [["xiaoxinyun1_01", "xiaoxinyun2_01"], ["xiaoxinyun1_02", "xiaoxinyun2_02"], ["zhiqingchun1_01", "zhiqingchun2_01"], ["zhiqingchun1_02", "zhiqingchun2_02"], ["wanghouyusheng1_01", "wanghouyusheng2_01"], ["wanghouyusheng1_02", "wanghouyusheng2_02"], ["wanghouyusheng1_03", "wanghouyusheng2_03"], \
        #                    ["lifyoftheparty1_01", "lifyoftheparty2_01"], ["lifyoftheparty1_02", "lifyoftheparty2_02"], ["lifyoftheparty1_03", "lifyoftheparty2_03"],\
        #                    ["dayu1_01", "dayu2_01"], ["dayu1_02", "dayu2_02"], ["qifengle1_01", "qifengle2_01"], ["qifengle1_02", "qifengle2_02"], ["zuimeideqidai1_01", "zuimeideqidai2_01"], \
        #                    ["zuimeideqidai1_02", "zuimeideqidai2_02"], ["saysomething1_01", "saysomething2_01"]]
        #
        # self.motion_name2 = ["1_2_faceZ", "2_faceZ", "3_clip1_faceZ", "3_clip2_faceZ", "4_faceZ", "5_faceZ",
        #                     "6_clip1_faceZ", "6_clip2_faceZ", "1_01_faceZ", "1_02_faceZ", "2_01_faceZ", "2_02_faceZ",
        #                      "2_03_faceZ", "3_01_faceZ", "3_02_faceZ", "3_03_faceZ"]
        # self.music_name2 = [["1AsIfItsYourLast"], ["2BBoomBBoom"], ["3CheerUp_01"], ["3CheerUp_02"], ["4solo"], ["5PlayingWithFire"],
        #                    ["6PeekABoo_01"], ["6PeekABoo_02"], ["create101_01"], ["create101_02"], ["OohAhh_01"],
        #                     ["OohAhh_02"], ["OohAhh_03"], ["ShapeofYou_01"], ["ShapeofYou_02"], ["ShapeofYou_03"]]

        self.motion_name = ["dance2_1_01_faceZ"]
        self.music_name = [["xiaoxinyun1_01", "xiaoxinyun2_01"]]
        self.motion_name2 = []
        self.music_name2 = [[]]

        self.qmean = self.get_qmean(train_path)
        self.zeros_channel = [31, 33, 35, 36, 46, 48, 52, 53, 55, 56, 61, 62, 64, 65, 70, 71, 73, 74, 79, 80, 82, 83, 91, 93, 95, 96, 106, 108, 112, 113, 115, 116, 121, 122, 124, 125, \
                              130, 131, 133, 134, 139, 140, 142, 143, 148, 150, 154, 156, 160, 162, 166, 168]
        self.hand_zero_channel = [31, 33, 35, 36]
        self.hand_zero_channel += [n for n in range(41, 86)]
        self.hand_zero_channel += [91, 93, 95, 96]
        self.hand_zero_channel += [n for n in range(101, 146)]
        self.hand_zero_channel += [148, 150, 154, 156, 160, 162, 166, 168]
        self.train_motion_data = []
        self.valid_motion_data = []
        self.train = []
        self.train_music_data = []
        self.train_music = []


        for i in range(len(self.motion_name)):
            start_ind = 0
            path_i = self.motion_name[i]

            motion_i, footcontact_i = self.get_motion(self.path, path_i)
            motion_i_end_position = np.load(self.path + path_i + "_end_position.npy")
            motion_i_end_position[np.isnan(motion_i_end_position)] = 0.0
            motion_i_and_endp = np.concatenate((motion_i, 0.001*motion_i_end_position), axis=1)
            motion_i_danceclass = np.zeros((motion_i.shape[0], self.hparams.dance_class))
            motion_i_danceclass[:, 0] = 1.

            motion_i_and_calss = [motion_i_and_endp, footcontact_i, motion_i_danceclass]
            self.train_motion_data.append(motion_i_and_calss)

            chroma_feature_i = []
            chroma_feature_i_length = None
            for music_j in range(len(self.music_name[i])):
                chroma_feature_i_j = self.get_onset_chroma_feature(music_wave_path + self.music_name[i][music_j])
                chroma_feature_i.append(chroma_feature_i_j)
                if chroma_feature_i_length is None:
                    chroma_feature_i_length = chroma_feature_i_j.shape[0]
                else:
                    chroma_feature_i_length = min(chroma_feature_i_length, chroma_feature_i_j.shape[0])

            self.train_music_data.append(chroma_feature_i)

            while (start_ind + 480) < motion_i.shape[0] and int((start_ind + 0.5) * 10. / 60.) + 80 < \
                    chroma_feature_i_length:
                for music_j in range(len(self.music_name[i])):
                    self.train.append((i, int(start_ind + 0.5)))
                    self.train_music.append((i, music_j, int((start_ind + 0.5) * 10. / 60.)))

                start_ind += self.moving_window
  
            print(len(self.train))
            print(len(self.train_music))

        last_motion_length = len(self.motion_name)

        for i in range(len(self.motion_name2)):
            start_ind = 0
            path_i = self.motion_name2[i]

            motion_i, footcontact_i = self.get_motion(self.path, path_i)
            motion_i_end_position = np.load(self.path + path_i + "_end_position.npy")
            motion_i_end_position[np.isnan(motion_i_end_position)] = 0.0
            motion_i_and_endp = np.concatenate((motion_i, 0.001*motion_i_end_position), axis=1)
            motion_i_danceclass = np.zeros((motion_i.shape[0], self.hparams.dance_class))
            motion_i_danceclass[:, 1] = 1.

            motion_i_and_calss = [motion_i_and_endp, footcontact_i, motion_i_danceclass]
            self.train_motion_data.append(motion_i_and_calss)

            chroma_feature_i = []
            chroma_feature_i_length = None
            for music_j in range(len(self.music_name2[i])):
                chroma_feature_i_j = self.get_onset_chroma_feature(music_wave_path + self.music_name2[i][music_j])
                chroma_feature_i.append(chroma_feature_i_j)
                if chroma_feature_i_length is None:
                    chroma_feature_i_length = chroma_feature_i_j.shape[0]
                else:
                    chroma_feature_i_length = min(chroma_feature_i_length, chroma_feature_i_j.shape[0])

            self.train_music_data.append(chroma_feature_i)
            while (start_ind + 480) < motion_i.shape[0] and int((start_ind + 0.5) * 10. / 60.) + 80 < \
                    chroma_feature_i_length:
                for music_j in range(len(self.music_name2[i])):
                    self.train.append((i+last_motion_length, int(start_ind + 0.5)))
                    self.train_music.append((i+last_motion_length, music_j, int((start_ind + 0.5) * 10. / 60.)))

                start_ind += self.moving_window
            print(len(self.train))
            print(len(self.train_music))


    def get_qmean(self, motion_paths):
        saved_data_path_qmean = self.path + "/all_qmean.npy"
        print(saved_data_path_qmean)
        if os.path.exists(saved_data_path_qmean):
            return np.load(saved_data_path_qmean)

        poses_r_all = []
        root_ry_all = []
        for i, path in enumerate(motion_paths):
            poses_r_arr, root_tran_arr, root_rx, root_ry = self.ani_process.read_poses(path)
            poses_r_all.append(copy.deepcopy(poses_r_arr))
            root_ry_all.append(copy.deepcopy(root_ry))

        np.save(self.path + "/all_ry.npy", np.array(root_ry_all))
        np.save(self.path + "/all_rot.npy", np.array(poses_r_all))
        all_quat_mean = self.ani_process.get_all_meanq(np.array(poses_r_all), np.array(root_ry_all))
        np.save(saved_data_path_qmean, all_quat_mean)
        return all_quat_mean

    def get_motion(self, path, path_i):
        saved_data_path = path + path_i + ".npy"
        saved_data_path_footcontact = path + path_i + "_foot.npy"
        if os.path.exists(saved_data_path) and os.path.exists(saved_data_path_footcontact):
            data = np.load(saved_data_path)
            data[np.isnan(data)] = 0.0
            motion_i = self.del_zeros(data)
            foot_contact_all = np.load(saved_data_path_footcontact)
            return motion_i, foot_contact_all

        poses_r_arr, root_tran_arr, root_rx, root_ry = self.ani_process.read_poses(path + path_i + ".sps")
        all_quat_mean = self.qmean
        face_angel_root_deform = self.ani_process.root_tran_deform(root_tran_arr, root_rx)
        root_rot_deform = self.ani_process.root_rot_deform(poses_r_arr[:, 0], root_ry, all_quat_mean[0])

        joint_rot_deform = self.ani_process.joint_mat2vector(poses_r_arr[:, 1:], all_quat_mean[1:])
        joint_rot_deform_reshape = joint_rot_deform.reshape(
            (joint_rot_deform.shape[0], joint_rot_deform.shape[1] * 3))

        motion_representation = np.concatenate((face_angel_root_deform, root_rot_deform, joint_rot_deform_reshape),
                                               axis=1)

        np.save(saved_data_path, motion_representation)

        foot_contact = np.load(saved_data_path_footcontact)
        return motion_representation, foot_contact




    def get_onset_chroma_feature(self, audio_path):
        chroma_feature_path = audio_path + "_12madmomchroma_beatonset_10fps.npy"
        return np.load(chroma_feature_path)

    def get_music_onset_chroma_feature(self, path):
        onset_chroma_path = path[:-4] + "_onset_chroma.npy"
        if os.path.exists(onset_chroma_path):
            return np.load(onset_chroma_path)
        dcp = madmom.audio.chroma.DeepChromaProcessor()

        chroma = dcp(path)

        proc = madmom.features.downbeats.RNNDownBeatProcessor()
        downbeats = proc(path)

        proc_onset = madmom.features.onsets.RNNOnsetProcessor()
        onset = proc_onset(path)

        downbeats_10fps_num = np.linspace(0, downbeats.shape[0] - 1, chroma.shape[0])
        inter_func = interp1d(np.arange(0, downbeats.shape[0]), downbeats, axis=0)
        downbeats_10fps = inter_func(downbeats_10fps_num)

        onset_10fps_num = np.linspace(0, onset.shape[0] - 1, chroma.shape[0])
        inter_func_onset = interp1d(np.arange(0, onset.shape[0]), onset, axis=0)
        onset_10fps = inter_func_onset(onset_10fps_num)

        chroma_beatonset_10fps = np.zeros((chroma.shape[0], 15))
        chroma_beatonset_10fps[:, :12] = chroma
        chroma_beatonset_10fps[:, 12:14] = downbeats_10fps[:, :2]
        chroma_beatonset_10fps[:, 14] = onset_10fps
        chroma_feature = chroma_beatonset_10fps
        np.save(onset_chroma_path, chroma_feature)
        return chroma_feature

    def motion_addnoise(self, motion):
        return motion + self.hparams.noise_para * np.random.randn(motion.shape[0], motion.shape[1])

    def del_zeros(self, motion):
        motion = np.delete(motion, self.zeros_channel, axis=1)
        return motion

    def add_zeros(self, motion):
        motion_out = np.zeros((motion.shape[0], motion.shape[1] + len(self.zeros_channel)))
        t = 0
        for ch in range(motion_out.shape[1]):
            if ch in self.zeros_channel:
                t += 1
            else:
                motion_out[:, ch] = motion[:, ch - t]

        return motion_out



    def recovery_motion_test(self, motion_clip, path):
        all_quat_mean = self.qmean
        print(motion_clip.shape)
        all_motion = self.add_zeros(motion_clip)

        root_trans_recovery = self.ani_process.root_tran_recovery(all_motion[:, :4])
        root_rot_recovery = self.ani_process.root_rot_recovery(all_motion[:, 4:7], all_motion[:, :4], all_quat_mean[0])

        motion_representation_joint = all_motion[:, 7:].reshape(
            (all_motion.shape[0], int((all_motion[:, 7:].shape[1]) / 3), 3))
        joint_rot_recovery = self.ani_process.joint_vector2mat(motion_representation_joint, all_quat_mean[1:])

        root_rot_recovery = np.expand_dims(root_rot_recovery, axis=1)
        rot_recovery = np.concatenate((root_rot_recovery, joint_rot_recovery), axis=1)
        write2sps(root_trans_recovery, rot_recovery, path)





    def __getitem__(self, index):
        file_id, start_ind = self.train[index]
        train_motion = self.train_motion_data[file_id][0]
        end_ind = start_ind + self.sequence_length
        motion_clip = copy.deepcopy(train_motion[start_ind:end_ind])
        foot_contact = self.train_motion_data[file_id][1]
        foot_contact_clip = copy.deepcopy(foot_contact[start_ind:end_ind])

        dance_class = self.train_motion_data[file_id][2]
        dance_class_clip = copy.deepcopy(dance_class[start_ind:end_ind])

        file_music_id, music_id, start_music_ind = self.train_music[index]
        end_music_ind = start_music_ind + self.sequence_chroma_length

        music_chroma_clip = copy.deepcopy(self.train_music_data[file_music_id][music_id][start_music_ind: end_music_ind])

        motion_clip_noise = self.motion_addnoise(motion_clip)

        return torch.from_numpy(motion_clip_noise).float(), torch.from_numpy(foot_contact_clip).float(), torch.from_numpy(dance_class_clip).float(), torch.from_numpy(music_chroma_clip).float()

    def get_test_data(self, music_wav_path, dance_type_=0):
        file_id, start_ind = self.train[0]
        valid_motion = copy.deepcopy(self.train_motion_data[file_id][0])
        motion_clip = valid_motion[:]
        valid_foot = copy.deepcopy(self.train_motion_data[file_id][1])
        foot_contact_clip = valid_foot[:]
        music_onset_chroma = self.get_music_onset_chroma_feature(music_wav_path)

        dance_type = np.zeros((6*music_onset_chroma.shape[0], self.hparams.dance_class))
        dance_type[:, dance_type_] = 1.
        return (torch.from_numpy(motion_clip).float()).unsqueeze(0), (torch.from_numpy(foot_contact_clip).float()).unsqueeze(0), (torch.from_numpy(dance_type).float()).unsqueeze(0), (torch.from_numpy(music_onset_chroma).float()).unsqueeze(0)



    def __len__(self):
        return len(self.train)













