import cv2
import numpy as np
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
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.load(row.path + "image.npy")*255.0
        mask = np.load(row.path + "label.npy")
        if self.transform:
            data = self.transform(image=image, mask=mask)
            image, mask = data["image"], data["mask"]
        return image, mask
    
    def __len__(self):
        return len(self.df)