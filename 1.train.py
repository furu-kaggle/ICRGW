import numpy as np
import pandas as pd
import os, glob, cv2, random, json
from tqdm import tqdm
import optuna
import matplotlib.pyplot as plt
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from segmentation_models_pytorch.encoders import get_preprocessing_params

class CFG:
    seed          = 101
    backbone      = 'timm-efficientnet-b0'
    encoder_weight= "noisy-student" #timm [imagenet / advprop / noisy-student]
    pretrain      = True
    pp_params     = get_preprocessing_params(backbone, pretrained=encoder_weight)
    img_size      = [256, 256]
    crop_size     = [256, 256]
    sub_img_size  = [256, 256]
    valid_size    = [256, 256]
    batch_size    = 64
    epochs        = 1
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
    weight_path   = None#"/kaggle/input/ic-rgwweights/model_checkpoint_e18.pt"
    inp_mode = 'bilinear'

train = pd.read_parquet("data/train.parquet")
valid = pd.read_parquet("data/validation.parquet")

pdf = pd.DataFrame(glob.glob("data/ashfloat32/*/"),columns=["path"])
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
        self.best_score = 0

    def __call__(self, trial):
        # suggest parameters
        self.CFG.lr = trial.suggest_float("lr", 5e-4, 8e-3,step=5e-4)
        self.CFG.alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.01)
        self.CFG.enc_ratio = trial.suggest_float("enc_ratio", 0.05, 0.15,step=0.01)
        self.CFG.lr_min = trial.suggest_float("lr_min", 5e-5,5e-4,step=1e-5)

        # create trainer with suggested parameters
        trainer = self.trainer_class(CFG=self.CFG, train=train, valid=valid)
        trainer.best_score = self.best_score

        # train model and get best score
        self.best_score = trainer.fit(epochs=self.CFG.epochs)

        return self.best_score

import datetime
# Get current date and time
now = datetime.datetime.now()

# Format as a string
now_str = now.strftime('%Y%m%d')

# Use in database file name
db_file = f'sqlite:///try_{now_str}.db'


# define optimization
def optimize(CFG, trainer_class, n_trials=100):
    objective = Objective(CFG, trainer_class)
    study = optuna.create_study(study_name=f'trial_{now_str}', storage=db_file, direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    study.trials_dataframe().to_csv(f'checkpoint/trials_{now_str}.csv', index=False)

    # Get parameter importances
    importances = optuna.importance.get_param_importances(study)
    # Convert to DataFrame and save
    pd.DataFrame(list(importances.items()), columns=['parameter', 'importance']).to_csv(f'checkpoint/importances_{now_str}.csv', index=False)

    # Save best params as json
    with open(f'checkpoint/best_params_{now_str}.json', 'w') as f:
        json.dump(study.best_params, f)

# use the function
optimize(CFG, Trainer, n_trials=5)