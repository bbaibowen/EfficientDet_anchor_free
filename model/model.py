import torch
from torch import nn
from .efficientnet import EfficientNet
from .bi import BiFpn
from .head import FCOSHead
from tool import postprocess
from .loss import FCOSLoss
class EfficientDet_Free(nn.Module):

    def __init__(self,config):
        super(EfficientDet_Free,self).__init__()
        self.config = config
        self.backbone = EfficientNet.from_pretrained(config.network)
        self.fpn = BiFpn(in_channels=self.backbone.get_list_features()[-5:],out_channels=config.out_channels,len_input=5,bi=3)
        self.head = FCOSHead(config.out_channels, config.n_class, config.n_conv, config.prior)
        self.postprocessor = postprocess.FCOSPostprocessor(
            config.threshold,
            config.top_n,
            config.nms_threshold,
            config.post_top_n,
            config.min_size,
            config.n_class,
        )
        self.loss = FCOSLoss(
            config.sizes,
            config.gamma,
            config.alpha,
            config.iou_loss_type,
            config.center_sample,
            config.fpn_strides,
            config.pos_radius,
        )
        self.fpn_strides = config.fpn_strides

    def compute_location(self, features):
        locations = []

        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, self.fpn_strides[i], feat.device
            )
            locations.append(location_per_level)

        return locations

    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2

        return location

    def forward(self, input, image_sizes=None, targets=None):
        features = self.backbone(input)[-5:]
        features = self.fpn(features)
        cls_pred, box_pred, center_pred = self.head(features)
        location = self.compute_location(features)

        if self.training:
            loss_cls, loss_box, loss_center = self.loss(
                location, cls_pred, box_pred, center_pred, targets
            )
            losses = {
                'loss_cls': loss_cls,
                'loss_box': loss_box,
                'loss_center': loss_center,
            }

            return None, losses

        else:
            boxes = self.postprocessor(
                location, cls_pred, box_pred, center_pred, image_sizes
            )

            return boxes, None



