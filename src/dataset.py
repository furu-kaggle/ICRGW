import cv2, random
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
        if self.mode=="train":
            if row.has_mask:
                sampling_timeid = 4
            else:
                sampling_timeid = random.randint(0,7)
        else:
            sampling_timeid = 4
        try:
            image = np.load(row[f"path{sampling_timeid}"] + "image.npy")*255.0
        except:
            print(row[f"path{sampling_timeid}"])
            image = np.load(row[f"path4"] + "image.npy")*255.0
        mask = np.load(row.path4 + "label.npy")
        
        if self.transform:
            data = self.transform(image=image, mask=mask)
            image, mask = data["image"], data["mask"].to(torch.float32)
        
        return image, mask
    
    def __len__(self):
        return len(self.df)
    

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