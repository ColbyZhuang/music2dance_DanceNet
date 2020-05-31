# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import torch
from torch import nn
import argparse
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from models.Optim import ScheduledOptim
from models.dancenet_rhythm_chroma import music2dance_model, GMMLogLoss
from data.multi_dance_prepare import dance_motion_music_pre
from utils.logger import Logger
from utils.misc import save_checkpoint
from utils.hparams import Hparams
from utils.osutils import *

hparams = Hparams()

is_train = False



def main():

    if not isdir(hparams.checkpoint):
        mkdir_p(hparams.checkpoint)
    if not isdir(hparams.vis_path):
        mkdir_p(hparams.vis_path)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False


    model = music2dance_model(hparams)
    model = model.cuda()
    #if len(hparams.device_ids) > 1:
    model = nn.DataParallel(model, device_ids=hparams.device_ids)
    criterion = GMMLogLoss(hparams.n_center, hparams.n_input_channels, hparams.sigma_bias)
    criterion_foot = torch.nn.BCELoss(size_average=True)
    criterion_motion_smooth = torch.nn.MSELoss(size_average=True)


    optimizer = ScheduledOptim(
        torch.optim.RMSprop(model.parameters(), eps=1e-08, weight_decay=1e-4),
        hparams.d_model, hparams.n_warmup_steps, hparams.init_lr)



    if isfile(hparams.resume):
        print("=> loading checkpoint '{}'".format(hparams.resume))
        check_point = torch.load(hparams.resume)
        start_epoch = check_point['epoch']
        #start_iter = 0
        model.load_state_dict(check_point['state_dict'])

        pretrained_dict = check_point['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # filter out unnecessary keys
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(hparams.resume, check_point['epoch']))
        logger = Logger(join(hparams.checkpoint, 'log.txt'), title="motion_wavenet", resume=True)
    else:
        print("=> no checkpoint found at '{}'".format(hparams.resume))
        start_epoch = 0
        logger = Logger(join(hparams.checkpoint, 'log.txt'), title="motion_wavenet")
        logger.set_names(['Epoch', 'LR', 'Train GMM Loss'])




    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters()) / 1000000.0))


    dataset = dance_motion_music_pre(train=True)
    train_dataset = torch.utils.data.DataLoader(dataset,num_workers=20,
        batch_size=hparams.batch_size, shuffle=True)

    if not is_train:
        test_music_path = "./test_music_result/music/wozuiqinaide.wav"
        motion, foot_contact, dance_class, music_onset_chroma = dataset.get_test_data(test_music_path, dance_type_=0)
        motion = motion.cuda()
        foot_contact = foot_contact.cuda()
        dance_class = dance_class.cuda()
        music_onset_chroma = music_onset_chroma.cuda()

        test_seq_input = motion[0, :1, :].unsqueeze(0).detach()
        test_seq_input_foot_contact = foot_contact[0, :1, :].unsqueeze(0).detach()
        with torch.no_grad():
            pred_motion, foot_cat = model.module.generation_long_dance(test_seq_input, test_seq_input_foot_contact,
                                                                      music_onset_chroma, dance_class)
        pred_motion_np = pred_motion[0].detach().cpu().numpy()
        path_train = "./test_music_result/result/wozuiqinaide_dance_smooth.sps"
        pred_motion_np = gaussian_filter1d(pred_motion_np, sigma=1, axis=0)
        dataset.recovery_motion_test(pred_motion_np[:, 0:117], path_train)

    else:
        for i in range(start_epoch, hparams.epoch_num):
            optimizer._update_learning_rate_normal(i)
            lr = optimizer.cur_lr
            model.train()
            # lr = adjust_learning_rate(optimizer, i, lr, [20000, 50000], 0.1)
            gmm_train_loss = train(train_dataset, model, optimizer, criterion, criterion_foot, criterion_motion_smooth)
            # writer.add_scalar('Train/Loss', gmm_train_loss.data, i+1)
            print("##" + "train epoch %03d, lr = %7f, loss = %5f" % (i, lr, gmm_train_loss))

            if i % 5 == 0:
                pred_val_motion = validate(train_dataset, model)
                output_seq = pred_val_motion.cpu().numpy()
                plot_data(output_seq, '{}/valid_epoch_{}'.format(hparams.vis_path, i))

                valid_motion_pred_np = pred_val_motion[0].detach().cpu().numpy()
                path_train = hparams.checkpoint + "/epoch" + str(i) + "_train.sps"
                dataset.recovery_motion_test(valid_motion_pred_np[:, 0:117], path_train)

                logger.append([i, lr, gmm_train_loss])
                save_checkpoint({
                    'epoch': i,
                    'state_dict': model.state_dict(),
                }, checkpoint=hparams.checkpoint, filename='checkpoint_%d.pth.tar' % (i))


def train(train_dataset, model, optimizer, criterion, criterion_foot, criterion_motion_smooth):
    model.train()
    total_loss = 0
    ind = 0
    total_n = len(train_dataset)
    for i, batch in enumerate(train_dataset):
        batch_motion_clip_noise, batch_foot_contact_clip, batch_dance_class_clip, music_chroma_clip = batch
        batch_motion_clip_noise = batch_motion_clip_noise.cuda()
        music_chroma_clip = music_chroma_clip.cuda()
        goal = batch_motion_clip_noise[:, 1:, :].contiguous()

        batch_foot_contact_clip = batch_foot_contact_clip.cuda()
        goal_foot_cat = batch_foot_contact_clip[:, 1:, :].contiguous()

        batch_dance_class_clip = batch_dance_class_clip.cuda()

        model.zero_grad()
        optimizer.zero_grad()
        pred_motion_dis, pred_motion_foot = model(batch_motion_clip_noise, batch_foot_contact_clip, music_chroma_clip,
                                                  batch_dance_class_clip)
        # (torch.autograd.Variable((output_joint_rot_seq[1:rot_length - 1] - output_joint_rot_seq[0:(rot_length - 2)]).data))
        loss_motion = criterion(pred_motion_dis, goal)
        loss_motion = loss_motion.mean()
        loss_foot = criterion_foot(pred_motion_foot, goal_foot_cat)

        # loss_mel_code_smooth = criterion_mel_code(mel_code[:, 1:], torch.autograd.Variable((mel_code[:, :-1]).data))
        start_mu = hparams.n_center
        end_mu = hparams.n_center + hparams.n_center * hparams.n_input_channels
        loss_motion_smooth = criterion_motion_smooth(
            pred_motion_dis[:, :-2, start_mu:end_mu] + pred_motion_dis[:, 2:, start_mu:end_mu],
            torch.autograd.Variable((2. * pred_motion_dis[:, 1:-1, start_mu:end_mu]).data))

        loss = loss_motion + 2. * loss_foot + 0.1 * loss_motion_smooth
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        total_loss += loss.item()
        ind += 1
        print("##" + "train itr %03d, motion_loss = %5f, motion_smooth = %5f, foot_loss = %5f" % (
        i + 1, loss_motion.item(), loss_motion_smooth.item(), loss_foot.item()))

    return total_loss / float(ind)


def validate(val_dataset, model):
    model.eval()
    for i, batch in enumerate(val_dataset):
        batch_motion_clip_noise, batch_foot_contact_clip, batch_dance_class_clip, music_onset_clip = batch
        batch_motion_clip_noise = batch_motion_clip_noise.cuda()
        batch_foot_contact_clip = batch_foot_contact_clip.cuda()
        batch_dance_class_clip = batch_dance_class_clip.cuda()

        music_onset_clip = music_onset_clip.cuda()
        test_seq_input = batch_motion_clip_noise[0, :1, :].unsqueeze(0).detach()
        test_seq_input_foot_contact = batch_foot_contact_clip[0, :1, :].unsqueeze(0).detach()
        test_seq_inpu_dance_class = batch_dance_class_clip[0, :, :].unsqueeze(0).detach()
        music_onset_clip_input = music_onset_clip[0, :, :].unsqueeze(0).detach()

        # test_for_code_seq = batch_motion_clip_noise[0, :, :].unsqueeze(0).detach()
        with torch.no_grad():
            pred_motion, foot_cat = model.module.generation_dance(test_seq_input, test_seq_input_foot_contact,
                                                                  music_onset_clip_input, test_seq_inpu_dance_class,
                                                                  batch_foot_contact_clip.shape[1] - 1)
        break

    return pred_motion



def plot_data(data,save_path='test.png'):
    plt.clf()
    plt.plot(data[0,:,11])
    plt.savefig(save_path)



if __name__ == '__main__':

    main()