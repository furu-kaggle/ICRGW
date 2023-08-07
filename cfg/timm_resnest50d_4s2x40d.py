from segmentation_models_pytorch.encoders import get_preprocessing_params
import torch

class CFG:
    seed          = 101
    backbone      = 'timm-resnest50d_4s2x40d'
    pretrain      = True
    encoder_weight= "imagenet" #"imagenet" #timm [imagenet / advprop / noisy-student]
    pp_params     = get_preprocessing_params(backbone, pretrained=encoder_weight)
    img_size      = [384, 384]
    crop_size     = [384, 384]
    sub_img_size  = [256, 256]
    valid_size    = [384, 384]
    epochs        = 60
    lr_epochs     = 60
    batch_size    = 8
    lr            = 0.004
    lr_min        = 5e-5
    enc_ratio     = 0.19
    weight_decay  = 0.01
    ema_decay     = 0.992
    fold          = 0
    num_classes   = 1
    alpha         = 0.1
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train         = True
    probe_thre    = True
    img_show      = False
    weight_path   = None
    inp_mode = 'bilinear'