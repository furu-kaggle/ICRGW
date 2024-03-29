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
from cfg.timm_resnest import CFG

df = pd.read_csv("data/train.csv")



pdf = pd.DataFrame(glob.glob("data/ashfloat32/*/"),columns=["path4"])
pdf["record_id"] = pdf.path4.apply(lambda x: x.split("/")[-2]).astype(int)
df = pd.merge(df,pdf,on=["record_id"])

for i in [0,1,2,3,5,6,7]:
    pdf = pd.DataFrame(glob.glob(f"data/ash{i}/*/"),columns=[f"path{i}"])
    pdf["record_id"] = pdf[f"path{i}"].apply(lambda x: x.split("/")[-2]).astype(int)
    df = pd.merge(df,pdf,on=["record_id"])

train = df[df.fold != 0].reset_index(drop=True)
valid = df[df.fold == 0].reset_index(drop=True)

from src import Trainer
class Objective:
    def __init__(self, CFG, trainer_class):
        self.CFG = CFG
        self.trainer_class = trainer_class
        self.best_score = 0

    def __call__(self, trial):
        # suggest parameters
        self.CFG.lr = trial.suggest_float("lr", 1e-3, 8e-3,step=1e-3)
        self.CFG.alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.05)
        self.CFG.enc_ratio = trial.suggest_float("enc_ratio", 0.05, 0.3,step=0.01)
        self.CFG.lr_min = trial.suggest_float("lr_min", 5e-5,5e-4,step=1e-5)

        # create trainer with suggested parameters
        trainer = self.trainer_class(CFG=self.CFG, train=train, valid=valid)
        #trainer.best_score = self.best_score

        # train model and get best score
        self.best_score = trainer.fit(epochs=self.CFG.epochs)
        # clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    try:
        # Try to load an existing study
        study = optuna.load_study(study_name=f'trial_{now_str}', storage=db_file)
        print("Loaded existing study.")
    except:
        # If study does not exist, create a new one
        study = optuna.create_study(study_name=f'trial_{now_str}', storage=db_file, direction="maximize")
        study.enqueue_trial(
            {
                'lr': CFG.lr, 
                'alpha': CFG.alpha, 
                'enc_ratio': CFG.enc_ratio, 
                'lr_min': CFG.lr_min
            }
        )
        print("Created new study.")
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
optimize(CFG, Trainer, n_trials=20)