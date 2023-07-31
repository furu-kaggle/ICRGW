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
from cfg.effb5_pesudo import CFG



df = pd.read_csv("data/train.csv")[["record_id","mask_sum","has_mask","fold","label_sum","dup_id","human_sum"]]
pdf = pd.DataFrame(glob.glob("data/pesudo3/**/**/**/*_pesudo.npy"),columns=["path"])
pdf["record_id"] = pdf.path.apply(lambda x: x.split("/")[-1].replace("_pesudo.npy","")).astype(int)
pdf = pdf.groupby("record_id").path.apply(list).reset_index()
df = pd.merge(df,pdf,on=["record_id"])

pdf = pd.DataFrame(glob.glob("data/ash3/*/image.npy"),columns=["img_path"])
pdf["record_id"] = pdf["img_path"].apply(lambda x: x.split("/")[-2]).astype(int)
df = pd.merge(df,pdf,on=["record_id"])

df1 = df.copy()

df = pd.read_csv("data/train.csv")[["record_id","mask_sum","has_mask","fold","label_sum","dup_id","human_sum"]]
pdf = pd.DataFrame(glob.glob("data/ash5/*/image.npy"),columns=["img_path"])
pdf["record_id"] = pdf["img_path"].apply(lambda x: x.split("/")[-2]).astype(int)
df = pd.merge(df,pdf,on=["record_id"])

pdf = pd.DataFrame(glob.glob("data/ash5/**/*_pesudo.npy"),columns=["path"])
pdf["record_id"] = pdf.path.apply(lambda x: x.split("/")[-1].replace("_pesudo.npy","")).astype(int)
pdf = pdf.groupby("record_id").path.apply(list).reset_index()
df = pd.merge(df,pdf,on=["record_id"])

df2 = df.copy()

df = pd.concat([df1,df2]).reset_index(drop=True)

vdf = pd.read_csv("data/train.csv",index_col=0)
pdf = pd.DataFrame(glob.glob("data/ashfloat32/*/"),columns=["path"])
pdf["record_id"] = pdf["path"].apply(lambda x: x.split("/")[-2]).astype(int)
vdf = pd.merge(vdf.drop(["path"],axis=1),pdf,on=["record_id"])
vdf["path_mask"] = vdf["path"] + "label.npy"

paths = [
    "base_checkpoint/model_checkpoint_0.683_0.pt",
    "base_checkpoint/model_checkpoint_0.686_1.pt",
    "base_checkpoint/model_checkpoint_0.678_2.pt",
    "base_checkpoint/model_checkpoint_0.685_3.pt",
]

from pesudo_src import Trainer

for fold, path in enumerate(paths):
    CFG.weight_path = path
    CFG.fold = fold
    train = df[df.fold == CFG.fold].reset_index(drop=True)
    valid = vdf[vdf.fold == CFG.fold].reset_index(drop=True)
    print(train)
    print(valid)
    best_score = Trainer(CFG=CFG, train=train, valid=valid).fit(epochs=CFG.epochs)