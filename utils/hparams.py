class Hparams(object):
    def __init__(self):
        # algorithm paramters
        #self.device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        self.device_ids = [0]
        self.itr_num = 1000000
        self.epoch_num = 1501
        self.batch_size = 128
        self.init_lr = 0.0004
        self.d_model = 256
        self.dropout_bound = [0.4]
        self.noise_para = 0.04
        self.data_float_scale = [0.5, 1.5]

        self.use_weigh_norm = False
        self.use_facebook = False
        self.n_warmup_steps = 500
        self.foot_concat_dim = 2
        self.foot_contact_embed = 10

        self.dance_class = 2
        self.dance_class_embed_dim = 5

        self.chroma_dim = 15

        # dancenet para.
        self.n_layers = 20
        self.n_maxdilation = 5
        self.n_center = 1
        self.n_input_channels = 117+15
        self.n_residual_channels = 64
        self.n_skip_channels = 512
        self.n_kernel_size = 2
        self.n_output_channels = self.n_center * (2 * self.n_input_channels + 1) + 2
        self.n_cond_channels = 30


        self.sigma_bias = 0.0

        self.n_use_cond = True
        self.n_use_global_cond = True

        # path
        self.data_path = "./data/motion_music_align_dance1anddance2/motion_align/"
        self.music_wave_path = "./data/motion_music_align_dance1anddance2/music_align/"



        self.checkpoint = "checkpoint/motion_dance1and2_music_chroma"


        self.resume = self.checkpoint + "/checkpoint_1500.pth.tar"
        #self.resume = ""


        self.vis_path = self.checkpoint + "/vis"
        self.result_path = self.checkpoint + "/result"





