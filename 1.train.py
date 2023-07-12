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
from cfg.effb1_ns import CFG

train = pd.read_parquet("data/train.parquet")
valid = pd.read_parquet("data/validation.parquet")

pdf = pd.DataFrame(glob.glob("data/ashfloat32/*/"),columns=["path4"])
pdf["record_id"] = pdf.path4.apply(lambda x: x.split("/")[-2])
train = pd.merge(train,pdf,on=["record_id"])
valid = pd.merge(valid, pdf,on=["record_id"])

for i in [0,1,2,3,5,6,7]:
    pdf = pd.DataFrame(glob.glob(f"data/ash{i}/*/"),columns=[f"path{i}"])
    pdf["record_id"] = pdf[f"path{i}"].apply(lambda x: x.split("/")[-2])
    train = pd.merge(train,pdf,on=["record_id"])

print(train)

from src import Trainer
best_score = Trainer(CFG=CFG, train=train, valid=valid).fit(epochs=CFG.epochs)