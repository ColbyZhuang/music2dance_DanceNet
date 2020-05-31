import torch
from torch import nn
from torch.nn import functional as F
import math


def sample_gmm(gmm_params, ncenter, ndim, sigma_bias1=0.0, sigma_bias2 = 0.0):
    assert ncenter > 0
    # you can set the sigma for unbiased or biased sampling, reference generating seqences with recurrent neural networks
    gmm_params_cpu = gmm_params.cpu().view(-1, (2 * ndim + 1) * ncenter)
    prob = nn.functional.softmax(gmm_params_cpu[:, :ncenter] * (1 + sigma_bias1), dim=1)
    sel_idx = torch.multinomial(prob, num_samples=1, replacement=True)
    mu = gmm_params_cpu[:, ncenter:ncenter + ncenter * ndim]
    # please note that use -logsigma
    sigma = torch.exp(-gmm_params_cpu[:, ncenter + ncenter * ndim:]) * sigma_bias2
    # set the selected sigma and mu from the gmm parameter sets
    sel_sigma = torch.empty(gmm_params_cpu.size(0), ndim).float()
    sel_mu = torch.empty(gmm_params_cpu.size(0), ndim).float()
    cur_sample = torch.randn(gmm_params_cpu.size(0), ndim).float()

    for b in range(gmm_params_cpu.size(0)):
        idx = sel_idx[b, 0]
        sel_sigma[b, :] = sigma[b, idx * ndim:(idx + 1) * ndim]
        sel_mu[b, :] = mu[b, idx * ndim:(idx + 1) * ndim]

    # sample with sel sigma and sel mean
    cur_sample = cur_sample * sel_sigma + sel_mu
    # cur_sample = sel_mu
    return cur_sample.unsqueeze(1).cuda()


"""
here we can compute the maximum receptive length used in the generative phase.
"""
def dancenet_max_fieldofvision(max_dilation, n_layers, kernel_size):
    dilations = []
    for i in range(n_layers):
        dilations.append(2 ** (i % max_dilation))

    field = 1 + kernel_size - 1
    for dilation in reversed(dilations):
        field += (kernel_size - 1) * dilation

    return field


class GMMLogLoss(nn.Module):
    def __init__(self, ncenter, ndim, sigma_bias=0.0, sigmamin=0.03):
        super(GMMLogLoss, self).__init__()
        self.ncenter = ncenter
        self.ndim = ndim
        self.sigma_bias = sigma_bias
        self.sigmamin = sigmamin

    """
        model_output: B x T x D1 (D1 = ncenter + ncenter*ndim*2)
        targets: B x T x ndim, the ground truth target data is shown here
        Implementation using logsumexp
        """

    def forward(self, model_output, targets):
        # split to logits here
        targets.requires_grad = False
        siz = targets.size()
        targets_rep = targets.unsqueeze(2).expand(siz[0], siz[1], self.ncenter, self.ndim)
        targets_rep.requires_grad = False
        # next we will requires the gradient
        logits = model_output[:, :, :self.ncenter]
        mus = model_output[:, :, self.ncenter:(self.ncenter + self.ncenter * self.ndim)].view(siz[0], siz[1], self.ncenter, self.ndim)
        # special design for the neg log sigmas
        neg_log_sigmas_out = model_output[:, :, (self.ncenter + self.ncenter * self.ndim):].view(siz[0], siz[1], self.ncenter, self.ndim)
        inv_sigmas_min = torch.ones(neg_log_sigmas_out.size()) * (1. / self.sigmamin)
        inv_sigmas_min = inv_sigmas_min.cuda()
        inv_sigmas_min_log = torch.log(inv_sigmas_min)
        neg_log_sigmas = torch.min(neg_log_sigmas_out, inv_sigmas_min_log)

        inv_sigmas = torch.exp(neg_log_sigmas) 

        loss_softmax = torch.logsumexp(logits, dim=-1)
        log_2pi = self.ndim * math.log(math.pi * 2) / 2.0

        # B x T x Ncenter x Ndim
        x = (targets_rep - mus) * inv_sigmas
        exp_tensor = logits + torch.sum(neg_log_sigmas, dim=-1) - log_2pi - 0.5 * torch.sum(x ** 2, dim=-1)
        # finally get the average loss over the batch or batch x time_length
        # note we must minimize the -log loss
        final_loss = loss_softmax - torch.logsumexp(exp_tensor, dim=-1)
        # remember to use mean operation outside for final_loss, including batch_size x max_time_length loss values, average over them
        # the non-pad mask is used if possible, that is to
        return final_loss.mean()


"""
add the weight normalization into the current layer
"""
def add_weight_normalization(layer):
    layer = torch.nn.utils.weight_norm(layer, name='weight')
    return layer


class CasualConv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, dilation=1, bias=True):
        super(CasualConv, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = nn.Conv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation, bias=bias)

    def weight_init(self):
        nn.init.xavier_normal(self.conv.weight)

    """
    remove the weight normalization here
    """
    def remove_weight_normalization(self):
        if hasattr(self.conv, 'weight_g'):
            self.conv = torch.nn.utils.remove_weight_norm(self.conv)

    def forward(self, x):
        padding = (int((self.kernel_size - 1) * self.dilation), 0)
        x = nn.functional.pad(x, padding)
        return self.conv(x)


class DilationBlock(nn.Module):
    def __init__(self, in_chs, out_chs, skip_chs, kernel_size, dilation, use_cond=True, use_g_cond=True, cond_chs=None, g_cond_chs=None,
                 need_transform=True, use_wn=False, use_facebook=False):
        super(DilationBlock, self).__init__()
        self.gate = CasualConv(in_chs, out_chs, kernel_size, dilation, True)
        self.filter = CasualConv(in_chs, out_chs, kernel_size, dilation, True)
        self.need_transform = need_transform
        self.use_cond = use_cond
        self.use_g_cond = use_g_cond

        conv_kernel_size = 1

        if use_cond is True:
            if use_wn:
                self.conv_gate = add_weight_normalization(nn.Conv1d(cond_chs, out_chs, conv_kernel_size, bias=False))
                self.conv_filter = add_weight_normalization(nn.Conv1d(cond_chs, out_chs, conv_kernel_size, bias=False))
            else:
                self.conv_gate = nn.Conv1d(cond_chs, out_chs, conv_kernel_size, bias=False)
                self.conv_filter = nn.Conv1d(cond_chs, out_chs, conv_kernel_size, bias=False)
        else:
            self.conv_gate = None
            self.conv_filter = None

        if use_g_cond is True:
            if use_wn:
                self.g_conv_gate = add_weight_normalization(nn.Conv1d(g_cond_chs, out_chs, conv_kernel_size, bias=False))
                self.g_conv_filter = add_weight_normalization(nn.Conv1d(g_cond_chs, out_chs, conv_kernel_size, bias=False))
            else:
                self.g_conv_gate = nn.Conv1d(g_cond_chs, out_chs, conv_kernel_size, bias=False)
                self.g_conv_filter = nn.Conv1d(g_cond_chs, out_chs, conv_kernel_size, bias=False)
        else:
            self.g_conv_gate = None
            self.g_conv_filter = None

        if self.need_transform:
            if use_wn:
                self.conv_transformed = add_weight_normalization(nn.Conv1d(out_chs, in_chs, conv_kernel_size))
            else:
                self.conv_transformed = nn.Conv1d(out_chs, in_chs, conv_kernel_size)
        else:
            self.conv_transformed = None

        if use_wn:
            self.conv_skip = add_weight_normalization(nn.Conv1d(out_chs, skip_chs, conv_kernel_size))
        else:
            self.conv_skip = nn.Conv1d(out_chs, skip_chs, conv_kernel_size)

    def weight_init(self):
        self.gate.weight_init()
        self.filter.weight_init()

        if self.use_cond is True:
            nn.init.xavier_normal(self.conv_gate)
            nn.init.xavier_normal(self.conv_filter)

        if self.need_transform:
            nn.init.xavier_normal(self.conv_transformed)
        nn.init.xavier_normal(self.conv_skip)

    """
     remove the weight normalization here
     """

    def remove_weight_normalization(self):
        self.gate.remove_weight_normalization()
        self.filter.remove_weight_normalization()
        if self.conv_gate is not None and hasattr(self.conv_gate, 'weight_g'):
            self.conv_gate = torch.nn.utils.remove_weight_norm(self.conv_gate)
        if self.conv_filter is not None and hasattr(self.conv_filter, 'weight_g'):
            self.conv_filter = torch.nn.utils.remove_weight_norm(self.conv_filter)

        if self.conv_transformed is not None and hasattr(self.conv_transformed, 'weight_g'):
            self.conv_transformed = torch.nn.utils.remove_weight_norm(self.conv_transformed)
        if hasattr(self.conv_skip, 'weight_g'):
            self.conv_skip = torch.nn.utils.remove_weight_norm(self.conv_skip)


    def forward(self, x, cond=None, g_cond=None):
        if self.use_cond is True and cond is None:
            raise RuntimeError("set using condition to true, but no cond tensor inputed")

        gx = self.gate(x)
        fx = self.filter(x)

        # add conditional
        if self.use_cond is True:
            gc = self.conv_gate(cond)
            fc = self.conv_filter(cond)
            gx = gx + gc
            fx = fx + fc

        # add global conditional
        if self.use_g_cond is True:
            g_gc = self.g_conv_gate(g_cond)
            g_fc = self.g_conv_filter(g_cond)
            gx = gx + g_gc
            fx = fx + g_fc

        out = torch.tanh(fx) * torch.sigmoid(gx)

        if self.need_transform:
            transformed = self.conv_transformed(out) + x
        else:
            transformed = None
        skip_out = self.conv_skip(out)

        return transformed, skip_out


class DilationSequence(nn.Module):
    def __init__(self, hparams):
        super(DilationSequence, self).__init__()

        dilation_blocks = []

        for i in range(hparams.n_layers):
            in_chs = hparams.n_residual_channels
            out_chs = hparams.n_residual_channels
            skip_chs = hparams.n_skip_channels
            ## kernel_size = hparam.n_kernel_size
            kernel_size = 2
            dilation = 2 ** (i % hparams.n_maxdilation)
            use_cond = hparams.n_use_cond
            use_g_cond = hparams.n_use_global_cond
            cond_chs = hparams.n_cond_channels
            g_cond_chs = hparams.dance_class * hparams.dance_class_embed_dim

            dilation_blocks.append(
                DilationBlock(in_chs, out_chs, skip_chs, kernel_size, dilation, use_cond=use_cond, use_g_cond=use_g_cond, cond_chs=cond_chs, g_cond_chs=g_cond_chs,
                              need_transform=(i != hparams.n_layers - 1), use_wn = hparams.use_weigh_norm,use_facebook = hparams.use_facebook))

        self.dilation_blocks = nn.ModuleList(dilation_blocks)

    def weight_init(self, init_method=None):
        for dilation_block in self.dilation_blocks:
            dilation_block.weight_init()

    """
    remove the weight normalization here
    """

    def remove_weight_normalization(self):
        blocks = nn.ModuleList()
        for model in self.dilation_blocks:
            model.remove_weight_normalization()
            blocks.append(model)
        self.dilation_blocks = blocks


    def forward(self, x, cond, g_cond):
        for i, dilation_block in enumerate(self.dilation_blocks):
            x, skip_out = dilation_block(x, cond, g_cond)

            if i == 0:
                output = skip_out
            else:
                output = output + skip_out
        return output


class PreProcessing(nn.Module):
    def __init__(self, hparams):
        super(PreProcessing, self).__init__()


        self.in_chs = hparams.n_input_channels + hparams.foot_contact_embed * hparams.foot_concat_dim
        out_chs = hparams.n_residual_channels
        self.conv_pre1 = nn.Conv1d(self.in_chs, out_chs, kernel_size=1, bias=True)
        self.conv_pre2 = nn.Conv1d(out_chs, out_chs, kernel_size=1, bias=True)

    def forward(self, x):

        x = x.transpose(1, 2)
        x = F.relu(self.conv_pre2(F.relu(self.conv_pre1(x))))
        return x


class PostProcessing(nn.Module):
    def __init__(self, hparams):
        super(PostProcessing, self).__init__()
        conv_kernel_size = 1
        self.conv_post_process = nn.Conv1d(hparams.n_skip_channels, hparams.n_skip_channels, conv_kernel_size,
                                           bias=True)
        self.conv_output = nn.Conv1d(hparams.n_skip_channels, hparams.n_output_channels, conv_kernel_size, bias=True)

    def forward(self, out):
        out = self.conv_post_process(F.relu(out))
        out = self.conv_output(F.relu(out))
        return out




class ConvStackChroma(nn.Module):
    def __init__(self, input_features, output_features):
        super(ConvStackChroma, self).__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, (2, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 24, (2, 1), stride=(1, 1)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 32, (2, 1), stride=(2, 1), padding=(1,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.bilstm = nn.LSTM(input_features*32, output_features, bidirectional=True)

    def forward(self, chroma):
        x = chroma.view(chroma.size(0), 1, chroma.size(1), chroma.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = x.flatten(-2)
        x, _ = self.bilstm(x)
        return x



class motion_WaveNet(nn.Module):
    def __init__(self, hparams):
        super(motion_WaveNet, self).__init__()
        self.hparams = hparams
        self.dropout_para = hparams.dropout_bound[0]
        self.dropout = nn.Dropout(self.dropout_para)
        self.embeds = nn.Embedding(hparams.foot_concat_dim, hparams.foot_contact_embed)

        self.prenet = PreProcessing(hparams)
        self.dilation_net = DilationSequence(hparams)
        self.postnet = PostProcessing(hparams)
        self.sigmoid = nn.Sigmoid()

        self.ncenter = hparams.n_center
        self.ndim = hparams.n_input_channels
        self.infer_length = dancenet_max_fieldofvision(hparams.n_maxdilation, hparams.n_layers, 2)

        #self.upsample = torch.nn.Upsample(scale_factor=512, mode='linear', align_corners=True)
    def set_dropout_para(self, para):
        self.dropout_para = para
        self.dropout = nn.Dropout(self.dropout_para)
    #def _upsampling(self, mel):
    #    return self.upsample(mel)

    """
    remove the weight normalization in the dancenet, this must be called before the inference
    """

    def remove_weight_normalization(self):
        self.dilation_net.remove_weight_normalization()

    def forward(self, in_seq, foot_contact_clip, cond=None, g_cond=None):
        if self.hparams.n_use_cond:
            cond_ = cond[:, :, 1:]
        else:
            cond_ = None

        if self.hparams.n_use_global_cond:
            g_cond_ = g_cond[:, :, 1:]
        else:
            g_cond_ = None

        x1 = in_seq[:, :-1, :].contiguous()
        x1 = self.dropout(x1)
        x2 = self.embeds(foot_contact_clip[:, :-1, :].long())
        x2_ = x2.reshape(x2.size(0), x2.size(1), x2.size(2) * x2.size(3))
        x = torch.cat((x1, x2_), dim=-1)
        x = self.prenet(x)
        x = self.dilation_net(x, cond_, g_cond_)
        x = self.postnet(x)
        x = x.transpose(1, 2)
        return x[:, :, :-2], self.sigmoid(x[:, :, -2:])

    def inference(self, x, cond=None):
        x = self.prenet(x)
        x = self.dilation_net(x, cond)
        x = self.postnet(x)
        return x

    def generate_sequences(self, tgt_seq, foot_contact, condition, g_condition, nframes=2, sigma2=0.1):
        # first to B x 1 x D if the tgt_seq is two dims
        if tgt_seq.ndimension() == 2:
            dec_motion_input = tgt_seq.unsqueeze(1)
            foot_contact_input = foot_contact.unsqueeze(1)
        else:
            dec_motion_input = tgt_seq.detach()
            foot_contact_input = foot_contact.detach()
        all_foot_contact = foot_contact_input.detach()
        x2 = self.embeds(foot_contact_input[:, :, :].long())
        x2_ = x2.reshape(x2.size(0), x2.size(1), x2.size(2) * x2.size(3))
        dec_input = torch.cat((dec_motion_input, x2_), dim=-1)
        dec_output = None

        for i in range(nframes):
            if dec_output is None:
                # note that the dec_output is B x D x T because of the convolution
                dec_output = self.prenet(dec_input)
                if self.hparams.n_use_cond:
                    cond = condition[:, :, 1:dec_output.size(2)+1]
                else:
                    cond = None

                if self.hparams.n_use_global_cond:
                    g_cond = g_condition[:, :, 1:dec_output.size(2)+1]
                else:
                    g_cond = None
            else:
                # add the last frame into the preprocess module
                # set the limited frame if the dec output exceed certain attention length, this is used for generating long sequences
                if dec_output.size(2) >= self.infer_length:
                    dec_output = torch.cat(
                        (dec_output[:, :, -self.infer_length + 1:], self.prenet(dec_input[:, -1, :].unsqueeze(1))),
                        dim=2)
                    if self.hparams.n_use_cond:
                        cond = condition[:, :, i + tgt_seq.size(1) + 1 - self.infer_length:i + tgt_seq.size(1) + 1]
                    else:
                        cond = None

                    if self.hparams.n_use_global_cond:
                        g_cond = g_condition[:, :, i + tgt_seq.size(1) + 1 - self.infer_length:i + tgt_seq.size(1) + 1]
                    else:
                        g_cond = None

                else:
                    dec_output = torch.cat((dec_output, self.prenet(dec_input[:, -1, :].unsqueeze(1))), dim=2)
                    if self.hparams.n_use_cond:
                        cond = condition[:, :, 1:dec_output.size(2) + 1]
                    else:
                        cond = None

                    if self.hparams.n_use_global_cond:
                        g_cond = g_condition[:, :, 1:dec_output.size(2) + 1]
                    else:
                        g_cond = None
            #print(dec_output.size())
            #print(cond.size())
            dec_hidden = dec_output.detach()
            dec_hidden = self.dilation_net(dec_hidden, cond, g_cond)
            #print(dec_hidden[0, 0])
            # get the last frame and sample from the GMM
            dec_hidden_all = self.postnet(dec_hidden[:, :, -1].unsqueeze(2))
            dec_hidden = dec_hidden_all[:, :-2, :]
            foot_cat = self.sigmoid(dec_hidden_all[:, -2:, :])
            # if i == 0:
            #     dec_hidden_all = dec_hidden
            # else:
            #     dec_hidden_all = torch.cat((dec_hidden_all, dec_hidden), dim=2)

            # sample from the GMM
            #print(dec_hidden[:, 0])
            new_input_motion = sample_gmm(dec_hidden, self.ncenter, self.ndim, sigma_bias1=3.0, sigma_bias2=sigma2)

            foot_cat = foot_cat.transpose(1, 2)

            bound_foot_cat = torch.ones_like(foot_cat) * 0.5
            zeros_foot_cat = torch.zeros_like(foot_cat)
            ones_foot_cat = torch.ones_like(foot_cat)
            foot_cat = where(foot_cat > bound_foot_cat, ones_foot_cat, zeros_foot_cat)
            all_foot_contact = torch.cat((all_foot_contact, foot_cat), 1)

            x2 = self.embeds(foot_cat.long())
            x2_ = x2.reshape(x2.size(0), x2.size(1), x2.size(2) * x2.size(3))
            new_input = torch.cat((new_input_motion, x2_), -1)
            # get the next decoder inputs
            dec_input = torch.cat((dec_input, new_input), dim=1)
        # return dec_input,dec_hiddens
        return dec_input[:, :, :117], all_foot_contact


def where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)




class music2dance_model(nn.Module):
    def __init__(self, hparams):
        super(music2dance_model, self).__init__()
        self.encode_cnn = ConvStackChroma(hparams.chroma_dim, hparams.n_cond_channels // 2)
        self.motion_wavenet = motion_WaveNet(hparams)
        self.upsample = nn.Upsample(scale_factor=(480./40.), mode='linear', align_corners=True)
        self.class_embeds = nn.Embedding(hparams.dance_class, hparams.dance_class_embed_dim)



    def forward(self, x, foot_contact, chroma, dance_class):
        chroma_code = self.encode_cnn(chroma)
        chroma_motion = self.upsample(chroma_code.transpose(1, 2))
        dance_class_embed_ = self.class_embeds(dance_class.long())
        dance_class_embed = dance_class_embed_.reshape(dance_class_embed_.size(0), dance_class_embed_.size(1), dance_class_embed_.size(2) * dance_class_embed_.size(3))

        pre_motion, pre_foot = self.motion_wavenet(x, foot_contact, cond=chroma_motion, g_cond=dance_class_embed.transpose(1, 2))

        return pre_motion, pre_foot

    def generation_dance(self, tgt_seq, foot_contact, chroma, dance_class, nframes=2, sigma2=0.1):
        chroma_code = self.encode_cnn(chroma)
        chroma_motion = self.upsample(chroma_code.transpose(1, 2))
        dance_class_embed_ = self.class_embeds(dance_class.long())
        dance_class_embed = dance_class_embed_.reshape(dance_class_embed_.size(0), dance_class_embed_.size(1), dance_class_embed_.size(2) * dance_class_embed_.size(3))

        pre_motion, pre_foot = self.motion_wavenet.generate_sequences(tgt_seq, foot_contact, chroma_motion, dance_class_embed.transpose(1, 2), nframes=nframes, sigma2=sigma2)
        return pre_motion, pre_foot


    # sliding windows to get context code, then get dance motion
    def generation_long_dance(self, tgt_seq, foot_contact, chroma, dance_class, sigma2=0.1):
        frame_num = 80
        number_clip = (chroma.shape[1] - 40) // 40
        last_clip_frame = chroma.shape[1] - number_clip * 40 - 40
        chroma_code = torch.zeros(30, (int(chroma.shape[1] * 480 / 80.+0.5)))
        for i in range(number_clip):
            if i == 0:
                code = self.encode_cnn(chroma[:, i*frame_num:(i+1)*frame_num, :])
                chroma_motion = self.upsample(code.transpose(1, 2))
                chroma_code[:, i * 480:(i + 1) * 480] = chroma_motion.clone().cpu()


            else:
                code = self.encode_cnn(chroma[:, i * 40:(i + 1) * 40+40, :])
                chroma_motion = self.upsample(code.transpose(1, 2))

                chroma_code[:, i * 240+120:i * 240 + 240] = (chroma_code[:, i * 240+120:i * 240 + 240] + chroma_motion[:, :, 120:240].clone().cpu())/2.
                chroma_code[:, i * 240 + 240:(i + 1) * 240 + 240] = chroma_motion[:, :, 240:].clone().cpu()

            if i == number_clip - 1 and last_clip_frame > 0:
                code = self.encode_cnn(chroma[:, -frame_num:, :])
                chroma_motion = self.upsample(code.transpose(1, 2))
                chroma_code[:, -480:((i + 1) * 240 + 240)] = (chroma_code[:, -480:((i + 1) * 240 + 240)] + chroma_motion[:, :, :(
                            480 - (chroma_code.shape[1] - ((i + 1) * 240 + 240)))].clone().cpu()) / 2.

                chroma_code[:, ((i + 1) * 240 + 240):] = chroma_motion[:,:, (480 - (chroma_code.shape[1] - ((i + 1) * 240 + 240))):].clone().cpu()

        chroma_code = chroma_code.cuda()
        chroma_code = chroma_code.unsqueeze(0).detach()

        dance_class_embed_ = self.class_embeds(dance_class.long())
        dance_class_embed = dance_class_embed_.reshape(dance_class_embed_.size(0), dance_class_embed_.size(1), dance_class_embed_.size(2) * dance_class_embed_.size(3))

        nframes = min(chroma_code.size(2) - 1, 2000)
        pre_motion, pre_foot = self.motion_wavenet.generate_sequences(tgt_seq, foot_contact, chroma_code, dance_class_embed.transpose(1, 2), nframes=nframes, sigma2=sigma2)
        return pre_motion, pre_foot

