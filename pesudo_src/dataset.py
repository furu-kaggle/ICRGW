import cv2, random
import numpy as np
import pandas as pd
import torch
import albumentations.pytorch
import albumentations as A


class maskDataset:
    def __init__(self, CFG, df, mode="train"):
        self.mode = mode
        self.transform = {
            "train": A.Compose([
                # A.ShiftScaleRotate(scale_limit=0.20, rotate_limit=0, shift_limit=0.1, p=0.25, border_mode=cv2.BORDER_CONSTANT, value=0),
                # A.GridDistortion(p=0.25),
                # A.HorizontalFlip(p=0.25),
                # A.VerticalFlip(p=0.25),
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

    def get_label(self, row):
        image = np.load(row[f"path4"] + "image.npy")*255.0
        mask = np.load(row.path4 + "label.npy")
        if (self.mode=="train")&(row.path != "nomask"):
            mask_h = np.load(row.path)
            mask_h = mask_h/row.human_sum * 0.5         
            mask = (mask + mask_h)
            mask[(mask >= 1.0)&(mask <= 1.0 + mask_h/2)] = 0.995
            mask = mask.clip(0.001,1)
        
        return image, mask
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.mode=="train":
            image = np.load(row[f"img_path"])*255.0
            mask =sum([np.load(path).transpose(1,2,0).astype(np.float32) for path in row[f"path"]])/len(row[f"path"])
        else:
            image = np.load(row[f"path"] + "image.npy")*255.0
            mask = np.load(row[f"path_mask"]).astype(np.float32)
            
        if self.transform:
            data = self.transform(image=image, mask=mask)
            image, mask = data["image"], data["mask"].to(torch.float16)

        hard_mask = mask.clone()
        return image.to(torch.float16), mask, hard_mask
    
    def __len__(self):
        return len(self.df)