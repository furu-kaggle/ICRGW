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

df = pd.read_csv("data/train.csv")
pdf = pd.DataFrame(glob.glob("data/ashfloat32/*/"),columns=["path4"])
pdf["record_id"] = pdf.path4.apply(lambda x: x.split("/")[-2]).astype(int)
df = pd.merge(df,pdf,on=["record_id"])

ndf = pd.DataFrame(glob.glob("data/ashfloat32/*/label_smooth.npy"),columns=["path"])
ndf["label"] = ndf.path.apply(lambda x: x.split("/")[-1].split("_")[-1].replace(".npy",""))
ndf["record_id"] = ndf.path.apply(lambda x: x.split("/")[-2]).astype(int)
df = df.drop(["path"],axis=1).merge(ndf,on=["record_id"],how="left")
df["path"].fillna("nomask",inplace=True)

# for i in [0,1,2,3,5,6,7]:
#     pdf = pd.DataFrame(glob.glob(f"data/ash{i}/*/"),columns=[f"path{i}"])
#     pdf["record_id"] = pdf[f"path{i}"].apply(lambda x: x.split("/")[-2]).astype(int)
#     df = pd.merge(df,pdf,on=["record_id"])

from src import Trainer
for fold in [0,1,2,3]:
    CFG.fold = fold
    train = df[df.fold != fold].reset_index(drop=True)
    valid = df[df.fold == fold].reset_index(drop=True)
    best_score = Trainer(CFG=CFG, train=train, valid=valid).fit(epochs=CFG.epochs)