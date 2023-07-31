import torch
from torch import nn
import segmentation_models_pytorch as smp

from typing import Optional, Union, List

from typing import Optional, Union, List
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)

class UNet(nn.Module):
    def __init__(self, CFG):
        super(UNet, self).__init__()
        self.CFG = CFG
        self.model = smp.Unet(
            encoder_name=CFG.backbone,     
            encoder_weights=CFG.encoder_weight if CFG.pretrain else None,
            in_channels=3,        
            classes=CFG.num_classes,
            activation=None
        )
        self.alpha = CFG.alpha
        if CFG.weight_path is not None:
            print(f"load pretrain model path:{CFG.weight_path}")
            self.model.load_state_dict(torch.load(CFG.weight_path, map_location=torch.device('cpu')))
            
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
        self.loss_fn_dist = nn.KLDivLoss(reduction="batchmean")
        
    def validate_forward(self, x):
        x = nn.functional.interpolate(x, size=self.CFG.valid_size,mode=self.CFG.inp_mode)
        x = self.model(x)
        x = nn.functional.interpolate(x, size=self.CFG.sub_img_size,mode=self.CFG.inp_mode)
        return x

    def forward(self, x, y=None, y_hard=None):
        x = self.model(x)
        if y is not None:
            soft_loss = (1-self.alpha)*self.loss_fn(x, y) + self.alpha*self.loss_fn_dice(x, y)
            #hard_loss = (1-self.alpha)*self.loss_fn(x, y_hard) + self.alpha*self.loss_fn_dice(x, y_hard)
            return soft_loss# + 0.3*hard_loss
        else:
            x = nn.functional.interpolate(x, size=self.CFG.sub_img_size,mode=self.CFG.inp_mode)
            return x
