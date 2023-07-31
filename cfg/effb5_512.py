from segmentation_models_pytorch.encoders import get_preprocessing_params
import torch

class CFG:
    seed          = 101
    backbone      = 'timm-efficientnet-b5'
    pretrain      = True
    encoder_weight= "noisy-student" #"imagenet" #timm [imagenet / advprop / noisy-student]
    pp_params     = get_preprocessing_params(backbone, pretrained=encoder_weight)
    img_size      = [512, 512]
    crop_size     = [512, 512]
    sub_img_size  = [256, 256]
    valid_size    = [512, 512]
    epochs        = 30
    lr_epochs     = 30
    batch_size    = 16
    lr            = 0.0025
    lr_min        = 9e-5
    enc_ratio     = 0.15
    weight_decay  = 0.01
    ema_decay     = 0.99
    fold          = 0
    num_classes   = 1
    alpha         = 0.15
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train         = True
    probe_thre    = True
    img_show      = False
    weight_path   = "checkpoint/model_base_0.684_0.pt"
    inp_mode = 'bilinear'