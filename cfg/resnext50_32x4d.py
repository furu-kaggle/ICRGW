from segmentation_models_pytorch.encoders import get_preprocessing_params
import torch

class CFG:
    seed          = 101
    backbone      = 'resnext50_32x4d'
    encoder_weight= "swsl" #timm [imagenet / advprop / noisy-student]
    pretrain      = True
    pp_params     = get_preprocessing_params(backbone, pretrained=encoder_weight)
    img_size      = [512, 512]
    crop_size     = [512, 512]
    sub_img_size  = [256, 256]
    valid_size    = [512, 512]
    batch_size    = 16
    epochs        = 30
    lr            = 0.0025
    lr_min        = 8e-05
    enc_ratio     = 0.1
    weight_decay  = 0.01
    ema_decay     = 0.99
    n_fold        = 5
    num_classes   = 1
    alpha         = 0.1
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train         = True
    probe_thre    = False
    img_show      = False
    weight_path   = None
    inp_mode = 'bilinear'