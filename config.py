class Config:

    def __init__(self):
        self.network = 'efficientnet-b0'
        self.local_rank = 0
        self.lr = 0.01
        self.l2 = 0.0001
        self.batch = 16
        self.epoch = 50
        self.n_save_sample = 5
        self.out_channel = 256
        self.n_class = 81
        self.prior = 0.01
        self.threshold = 0.05
        self.top_n = 1000
        self.nms_threshold = 0.6
        self.post_top_n = 100
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.gamma = 2.0
        self.alpha = 0.25
        self.iou_loss_type = 'giou'
        self.pos_radius = 1.5
        self.center_sample = True
        self.sizes = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]]
        self.train_min_size_range = (512,640)
        self.train_max_size = 800
        self.test_min_size = 512
        self.test_max_size = 800
        self.pixel_mean = [0.40789654, 0.44719302, 0.47026115]
        self.pixel_std = [0.28863828, 0.27408164, 0.27809835]
        self.size_divisible = 32
        self.min_size = 0

