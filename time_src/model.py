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

class UNet(SegmentationModel):
    def __init__(
        self,
        CFG,
        encoder_name: str = 'timm-efficientnet-b5',
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "noisy-student",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        expand_dim=1
    ):
        super(UNet, self).__init__()
        self.CFG = CFG
        encoder_name=CFG.backbone
        encoder_weights=CFG.encoder_weight if CFG.pretrain else None
        in_channels=3
        classes=CFG.num_classes

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=CFG.encoder_weight if CFG.pretrain else None,
        )
        self.decoder = UnetDecoder(
            encoder_channels=tuple(i * expand_dim for i in self.encoder.out_channels),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1]*expand_dim, **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
        self.alpha = CFG.alpha

    def get_mask(self, x_ashs):
        #f1 = self.encoder(x_ashs[:,0,:,:,:])
        f2 = self.encoder(x_ashs[:,1,:,:,:])
        f3 = self.encoder(x_ashs[:,2,:,:,:])
        f4 = self.encoder(x_ashs[:,3,:,:,:])
        #f5 = self.encoder(x_ashs[:,4,:,:,:])
        features = []
        #for f1t, f2t, f3t, f4t, f5t in zip(f1, f2, f3, f4, f5):
        for f2t, f3t, f4t in zip(f2, f3, f4):
            #finput = 0.05*f1t + 0.1*f2t + 0.7*f3t + 0.1*f4t + 0.05*f5t
            finput = 0.1*f2t + 0.8*f3t + 0.1*f4t
            features.append(finput)
            
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        return masks

    def validate_forward(self, x):
        #x = nn.functional.interpolate(x, size=self.CFG.valid_size,mode=self.CFG.inp_mode)
        x = self.get_mask(x)
        x = nn.functional.interpolate(x, size=self.CFG.sub_img_size,mode=self.CFG.inp_mode)
        return x
        
    def forward(self, x_ashs, labels=None):
        masks = self.get_mask(x_ashs)
        if labels is not None:
            loss = (1-self.alpha)*self.loss_fn(masks, labels) + self.alpha*self.loss_fn_dice(masks, labels)
            return loss
        else:
            x = nn.functional.interpolate(x, size=self.CFG.sub_img_size,mode=self.CFG.inp_mode)
            return masks