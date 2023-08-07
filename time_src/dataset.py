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
                A.Resize(*CFG.sub_img_size),
                A.Normalize(mean=CFG.pp_params["mean"],std=CFG.pp_params["std"]),
                A.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ], p=1.0),

            "test": A.ReplayCompose([
                A.Resize(*CFG.sub_img_size),
                A.Normalize(mean=CFG.pp_params["mean"],std=CFG.pp_params["std"]),
                A.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ], p=1.0)
        }[mode]
        self.df = df
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        head_img1 = np.load(row.path1)*255.0
        head_img2 = np.load(row.path3)*255.0
        image = np.load(row.path4 + "image.npy")*255.0
        back_img1 = np.load(row.path5)*255.0
        back_img2 = np.load(row.path7)*255.0
        mask = np.load(row.path4 + "label.npy")
        if (self.mode=="train")&(row.path != "nomask"):
            mask_h = np.load(row.path)
            mask_h = mask_h/row.human_sum * 0.5
            mask = (mask + mask_h).clip(0,1)
        
        data = self.transform(image=image, mask=mask)
        image, mask, replay = data["image"], data["mask"].to(torch.float32), data['replay']
        head_img1 = A.ReplayCompose.replay(replay, image=head_img1)["image"]
        head_img2 = A.ReplayCompose.replay(replay, image=head_img2)["image"]
        back_img1 = A.ReplayCompose.replay(replay, image=back_img1)["image"]
        back_img2 = A.ReplayCompose.replay(replay, image=back_img2)["image"]
        image = torch.stack([head_img1 ,head_img2, image, back_img1, back_img2])
        return image.to(torch.float32), mask
    
    def __len__(self):
        return len(self.df)