import torch
from torch import nn
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    def __init__(self, CFG):
        super(UNet, self).__init__()
        self.CFG = CFG
        self.model = smp.UnetPlusPlus(
            encoder_name=CFG.backbone,     
            encoder_weights=CFG.encoder_weight if CFG.pretrain else None,
            in_channels=3,        
            classes=CFG.num_classes,
            activation=None,
        )
        self.alpha = CFG.alpha
        if CFG.weight_path is not None:
            print(f"load pretrain model path:{CFG.weight_path}")
            self.model.load_state_dict(torch.load(CFG.weight_path, map_location=torch.device('cpu')))
            
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
        
    def validate_forward(self, x):
        x = nn.functional.interpolate(x, size=self.CFG.valid_size,mode=self.CFG.inp_mode)
        x = self.model(x)
        x = nn.functional.interpolate(x, size=self.CFG.sub_img_size,mode=self.CFG.inp_mode)
        return x

    def forward(self, x, y=None):
        x = self.model(x)
        if y is not None:
            loss = (1-self.alpha)*self.loss_fn(x, y) + self.alpha*self.loss_fn_dice(x, y)
            return loss
        else:
            x = nn.functional.interpolate(x, size=self.CFG.sub_img_size,mode=self.CFG.inp_mode)
            return x