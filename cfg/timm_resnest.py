from segmentation_models_pytorch.encoders import get_preprocessing_params
import torch

class CFG:
    seed          = 101
    backbone      = 'timm-resnest50d'
    encoder_weight= "imagenet" #timm [imagenet / advprop / noisy-student]
    pretrain      = True
    pp_params     = get_preprocessing_params(backbone, pretrained=encoder_weight)
    img_size      = [384, 384]
    crop_size     = [384, 384]
    sub_img_size  = [256, 256]
    valid_size    = [384, 384]
    batch_size    = 16
    epochs        = 30
    lr_epochs     = 30
    lr            = 0.0025
    lr_min        = 9e-5
    enc_ratio     = 0.15
    weight_decay  = 0.01
    ema_decay     = 0.99
    fold          = 0
    num_classes   = 1
    alpha         = 0.12
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train         = True
    probe_thre    = False
    img_show      = False
    weight_path   = None
    inp_mode = 'bilinear'