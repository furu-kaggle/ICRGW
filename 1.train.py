import numpy as np
import pandas as pd
import os, glob, cv2, random, json
from tqdm import tqdm
import optuna
import matplotlib.pyplot as plt
import torch    
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.set_flush_denormal(True)

from segmentation_models_pytorch.encoders import get_preprocessing_params

class CFG:
    seed          = 101
    backbone      = 'timm-efficientnet-b0'
    encoder_weight= "noisy-student" #timm [imagenet / advprop / noisy-student]
    pretrain      = True
    pp_params     = get_preprocessing_params(backbone, pretrained=encoder_weight)
    img_size      = [512, 512]
    crop_size     = [448, 448]
    sub_img_size  = [256, 256]
    valid_size    = [512, 512]
    batch_size    = 16
    epochs        = 20
    lr            = 7.5e-3
    lr_min        = 1e-4
    enc_ratio     = 0.01
    weight_decay  = 0.01
    ema_decay     = 0.99
    n_fold        = 5
    num_classes   = 1
    alpha         = 0.25
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train         = True
    probe_thre    = False
    img_show      = False
    weight_path   = None#"/kaggle/input/ic-rgwweights/model_checkpoint_e18.pt"
    inp_mode = 'bilinear'

train = pd.read_parquet("data/ic-rgw-basic-eda/train.parquet")
valid = pd.read_parquet("data/ic-rgw-basic-eda/validation.parquet")

#valid = valid[valid.has_mask.astype(bool)].reset_index(drop=True)

pdf = pd.DataFrame(glob.glob("/kaggle/input/ic-rgwbaseline-simple-rgw/*/"),columns=["path"])
pdf["record_id"] = pdf.path.apply(lambda x: x.split("/")[-2])
train = pd.merge(train,pdf,on=["record_id"])
valid = pd.merge(valid,pdf,on=["record_id"])

from src import Trainer
#debug
#trainer = Trainer(CFG=CFG,train=train,valid=valid)
#best_score = trainer.fit(epochs=CFG.epochs)
#model = trainer.model.eval()

class Objective:
    def __init__(self, CFG, trainer_class):
        self.CFG = CFG
        self.trainer_class = trainer_class

    def __call__(self, trial):
        # suggest parameters
        self.CFG.lr = trial.suggest_float("lr", 5e-4, 8e-3,step=5e-4)
        self.CFG.alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.01)
        self.CFG.enc_ratio = trial.suggest_float("enc_ratio", 0.05, 0.15,step=0.01)
        self.CFG.lr_min = trial.suggest_float("lr_min", 5e-5,5e-4,step=1e-5)

        # create trainer with suggested parameters
        trainer = self.trainer_class(CFG=self.CFG, train=train, valid=valid)

        # train model and get best score
        best_score = trainer.fit(epochs=self.CFG.epochs)

        return best_score

import datetime
# Get current date and time
now = datetime.datetime.now()

# Format as a string
now_str = now.strftime('%Y%m%d_%H%M%S')

# Use in database file name
db_file = f'sqlite:///try_{now_str}.db'


# define optimization
def optimize(CFG, trainer_class, n_trials=100):
    objective = Objective(CFG, trainer_class)
    study = optuna.create_study(study_name=f'trial_{now_str}', storage=db_file, direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Plotting importance
    fig, ax = plt.subplots()
    optuna.visualization.plot_param_importances(study, ax=ax)
    plt.savefig(f'checkpoint/importances_{now_str}.png')

    # Save best params as json
    with open(f'checkpoint/best_params_{now_str}.json', 'w') as f:
        json.dump(study.best_params, f)

# use the function
optimize(CFG, Trainer, n_trials=10)