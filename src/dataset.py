import cv2, random
import numpy as np
import pandas as pd
import torch
import albumentations.pytorch
import albumentations as A


class maskDataset:
    def __init__(self, df, CFG, mode="train"):
        self.mode = mode
        self.transform = {
            "train": A.Compose([
                A.ShiftScaleRotate(scale_limit=0.20, rotate_limit=0, shift_limit=0.1, p=0.25, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.GridDistortion(p=0.25),
                A.HorizontalFlip(p=0.25),
                A.VerticalFlip(p=0.25),
                A.Resize(*CFG.img_size),
                A.RandomCrop(*CFG.crop_size),
                A.Normalize(mean=CFG.pp_params["mean"],std=CFG.pp_params["std"]),
                A.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ], p=1.0),

            "valid": A.Compose([
                A.Resize(*CFG.sub_img_size),
                A.Normalize(mean=CFG.pp_params["mean"],std=CFG.pp_params["std"]),
                A.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ], p=1.0),

            "test": A.Compose([
                A.Resize(*CFG.sub_img_size),
                A.Normalize(mean=CFG.pp_params["mean"],std=CFG.pp_params["std"]),
                A.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ], p=1.0)
        }[mode]
        self.df = df
        if self.mode=="train":
           df_mask = self.df[self.df.has_mask.astype(bool)]
           df_nomask = self.df[~self.df.has_mask.astype(bool)]
           df_nomask = df_nomask[df_nomask.label_sum==0]
           self.df = pd.concat([df_mask,df_nomask]).reset_index(drop=True)
        self.dup_ids = self.df.dup_id.unique()

        #self.mixup_cand = self.df.groupby("fold_dup_id")["dup_id"].apply(list).to_dict()

    def load_data(self, row):
        sampling_timeid = 4
        image = np.load(row[f"path{sampling_timeid}"] + "image.npy")*255.0
        mask = np.load(row.path4 + "label.npy")
        if (self.mode=="train")&(row.path != "nomask"):
            mask_h = np.load(row.path)
            mask_h = mask_h/row.human_sum * 0.5
            mask = (mask + mask_h).clip(0,1)
        if self.transform:
            data = self.transform(image=image, mask=mask)
            image, mask = data["image"], data["mask"].to(torch.float32)
        
        return image, mask
        
    def __getitem__(self, idx):
        dup_idx = self.dup_ids[idx]
        rows = self.df[self.df.dup_id == dup_idx]
        image, mask = self.load_data(rows.sample(n=1).iloc[0])
        return image, mask
    
    def __len__(self):
        return len(self.dup_ids)
    

class timemaskDataset:
    def __init__(self, df, CFG, mode="train"):
        self.mode = mode
        self.transform = {
            "train": A.ReplayCompose([
                A.ShiftScaleRotate(scale_limit=0.20, rotate_limit=0, shift_limit=0.1, p=0.25, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.HorizontalFlip(p=0.25),
                A.VerticalFlip(p=0.25),
                A.Resize(*CFG.img_size),
                A.RandomCrop(*CFG.crop_size),
                A.Normalize(mean=CFG.pp_params["mean"],std=CFG.pp_params["std"]),
                A.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ], p=1.0),

            "valid": A.ReplayCompose([
                A.Resize(*CFG.img_size),
                A.Normalize(mean=CFG.pp_params["mean"],std=CFG.pp_params["std"]),
                A.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ], p=1.0),

            "test": A.ReplayCompose([
                A.Resize(*CFG.img_size),
                A.Normalize(mean=CFG.pp_params["mean"],std=CFG.pp_params["std"]),
                A.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ], p=1.0)
        }[mode]
        self.df = df
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        head_img = np.load(row.path3 + "image.npy")*255.0
        image = np.load(row.path + "image.npy")*255.0
        back_img = np.load(row.path5 + "image.npy")*255.0
        mask = np.load(row.path + "label.npy")
        
        data = self.transform(image=image, mask=mask)
        image, mask, replay = data["image"], data["mask"].to(torch.float32), data['replay']
        head_img = A.ReplayCompose.replay(replay, image=head_img)["image"]
        back_img = A.ReplayCompose.replay(replay, image=back_img)["image"]
        image = torch.stack([head_img, image, back_img])
        return image.to(torch.float32), mask
    
    def __len__(self):
        return len(self.df)