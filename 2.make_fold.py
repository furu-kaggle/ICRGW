import numpy as np
import pandas as pd
import os, glob, cv2, random, json
from tqdm import tqdm
import optuna
import matplotlib.pyplot as plt
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from segmentation_models_pytorch.encoders import get_preprocessing_params
from cfg.effb5_time import CFG

df = pd.read_csv("data/train.csv").drop(["path"],axis=1)
pdf = pd.DataFrame(glob.glob("data/ashfloat32/*/"),columns=["path4"])
pdf["record_id"] = pdf.path4.apply(lambda x: x.split("/")[-2]).astype(int)
df = pd.merge(df,pdf,on=["record_id"])

ndf = pd.DataFrame(glob.glob("data/ashfloat32/*/label_smooth_*.npy"),columns=["path"])
ndf["record_id"] = ndf.path.apply(lambda x: x.split("/")[-2]).astype(int)
df = df.merge(ndf,on=["record_id"],how="left")
df["path"].fillna("nomask",inplace=True)

print(df["path"])

for i in [1,3,5,7]:
    pdf = pd.DataFrame(glob.glob(f"data/ash{i}/*/image.npy"),columns=[f"path{i}"])
    pdf["record_id"] = pdf[f"path{i}"].apply(lambda x: x.split("/")[-2]).astype(int)
    df = pd.merge(df,pdf,on=["record_id"])

from time_src import Trainer
for fold in [0]:
    CFG.fold = fold
    train = df[df.fold != fold].reset_index(drop=True)
    valid = df[df.fold == fold].reset_index(drop=True)
    best_score = Trainer(CFG=CFG, train=train, valid=valid).fit(epochs=CFG.epochs)