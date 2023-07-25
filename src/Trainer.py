import os, glob
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import torch.optim as optim

import timm
import timm.scheduler

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.modules.batchnorm import _BatchNorm

from .model import UNet, MultiTimeUNet
from .dataset import maskDataset, timemaskDataset




class Trainer:
    def __init__(self, CFG, train, valid):
        self.CFG = CFG
        self.validation_losses = []
        self.epoch_losses = []
        self.learning_rates = []
        self.cumulative_mask_pred = []
        self.cumulative_mask_true = []
        model = UNet(CFG=CFG).to(CFG.device)
        try:
           import torch._dynamo
           torch._dynamo.reset()
           self.model = torch.compile(model, mode="max-autotune")
        except:
            print("torch version < 2.0.0 so we don't apply torch.compile ")
            self.model = model
        self.ema_model = timm.utils.ModelEmaV2(self.model, decay=CFG.ema_decay)
        group_decay_encoder, group_no_decay_encoder = self.group_weight(self.model.model.encoder)
        group_decay_decoder, group_no_decay_decoder = self.group_weight(self.model.model.decoder)
        self.optimizer = optim.AdamW([
            {'params': group_decay_encoder, 'lr': CFG.lr * CFG.enc_ratio},
            {'params': group_no_decay_encoder, 'lr': CFG.lr * CFG.enc_ratio, 'weight_decay':0.0},
            {'params': group_decay_decoder},
            {'params': group_no_decay_decoder, 'weight_decay':0.0}
        ], lr=CFG.lr, weight_decay=CFG.weight_decay)
        dataset_train = maskDataset(df=train,CFG=CFG)
        dataset_validation = maskDataset(df=valid,CFG=CFG,mode="valid")

        self.data_loader_train = DataLoader(
            dataset_train, 
            batch_size=CFG.batch_size, 
            shuffle=True,
            pin_memory=True,
            drop_last=True, 
            num_workers=min(CFG.batch_size, os.cpu_count())
        )
        self.data_loader_validation = DataLoader(
            dataset_validation, 
            batch_size=16, 
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=8
        )
        self.lr_scheduler = timm.scheduler.CosineLRScheduler(
            self.optimizer,
            t_initial=CFG.epochs*len(self.data_loader_train),
            lr_min=CFG.lr_min,
            t_in_epochs=True,
        )
        self.best_score = 0
        
    def get_threshold(self, e):
        self.model.eval()
        self.cumulative_mask_pred = []
        self.cumulative_mask_true = []
        losses = []
        with torch.no_grad():
            pbar = tqdm(enumerate(self.data_loader_validation),total=len(self.data_loader_validation))
            for i, (images, mask) in pbar:
                images = images.to(self.CFG.device,dtype=torch.float32,non_blocking=True)
                mask = mask.to(self.CFG.device,dtype=torch.float32,non_blocking=True)
                mask_pred = self.model.validate_forward(images)
                loss = self.model.loss_fn(mask_pred, mask)
                losses.append(loss.item())
                self.cumulative_mask_pred.append(mask_pred.sigmoid())
                self.cumulative_mask_true.append(mask)
                pbar.set_description("[loss %f]" % (loss))
        print("calc dice score....")
        self.cumulative_mask_pred = torch.cat(self.cumulative_mask_pred, dim=0).flatten()
        self.cumulative_mask_true = torch.cat(self.cumulative_mask_true, dim=0).flatten()
        thresholds_to_test = [round(x * 0.01, 2) for x in range(101)]
        optim_threshold = 0.975
        best_dice_score = -1

        thresholds = []
        dice_scores = []
        for t in thresholds_to_test:
            dice_score = self.test_threshold(t)
            if dice_score > best_dice_score:
                best_dice_score = dice_score
                optim_threshold = t

            thresholds.append(t)
            dice_scores.append(dice_score)

        print(f'Best Threshold: {optim_threshold} with dice: {best_dice_score}')
        df_threshold_data = pd.DataFrame({'Threshold': thresholds, 'Dice Score': dice_scores})
        avg_loss = torch.Tensor(losses).mean().item()
        self.validation_losses.append(avg_loss)
        sns.lineplot(data=df_threshold_data, x='Threshold', y='Dice Score')
        plt.axhline(y=best_dice_score, color='green')
        plt.axvline(x=optim_threshold, color='green')
        plt.text(-0.02, best_dice_score * 0.96, f'{best_dice_score:.3f}', va='center', ha='left', color='green')
        plt.text(optim_threshold - 0.01, 0.02, f'{optim_threshold}', va='center', ha='right', color='green')
        plt.ylim(bottom=0)
        plt.title('Threshold vs Dice Score')
        plt.savefig(f'checkpoint/thre-dice-relation_{best_dice_score:.3f}.png')
        plt.clf()
        print("Validation loss after", round(avg_loss, 4))
        return best_dice_score

        
    def group_weight(self, module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.Conv2d):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, _BatchNorm):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        return group_decay, group_no_decay
        
    def dice_score(self, inputs, targets, smooth=1):        
        intersection = (inputs.view(-1) * targets.view(-1)).sum()
        dice = (2.0 *intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return dice
    
    def test_threshold(self, threshold: float) -> float:
        after_threshold = torch.zeros_like(self.cumulative_mask_pred, device=self.CFG.device)
        after_threshold[self.cumulative_mask_pred > threshold] = 1
        after_threshold[self.cumulative_mask_pred < threshold] = 0
        score = self.dice_score(self.cumulative_mask_true, after_threshold.flatten()).item()
        return score

    def fit(self,
            epochs: int = 10,
            eval_every: int = 1,
            ):
  
        for e in range(epochs):
            total_loss = 0
            total_nums = 0
            scaler = torch.cuda.amp.GradScaler()
            pbar = tqdm(enumerate(self.data_loader_train),total=len(self.data_loader_train))
            self.model.train()
            for i, (images, mask) in pbar:
                images = images.to(self.CFG.device,dtype=torch.float32,non_blocking=True)
                mask = mask.to(self.CFG.device,dtype=torch.float32,non_blocking=True)
                
                with autocast():
                    loss = self.model(images, mask)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                self.ema_model.update(self.model)

                total_loss += (loss.detach().item() * mask.size(0))
                total_nums += mask.size(0)
                
                pbar.set_description("[loss %f, lr %e]" % (total_loss / total_nums, self.optimizer.param_groups[0]['lr']))

                # Adjusts learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(e*len(self.data_loader_train)+i)

            # Reports on the path
            self.epoch_losses.append(total_loss/total_nums)
            print('Train Epoch: {} Average Loss: {:.6f}'.format(e, total_loss/total_nums))

            if e >= 24:
                best_dice_score = self.get_threshold(e)
                if self.best_score < best_dice_score:
                    self.best_score = best_dice_score
                    torch.save(self.model.model.state_dict(), f"checkpoint/model_checkpoint_{best_dice_score:.3f}_{self.CFG.fold}.pt")
                    for path in sorted(glob.glob(f"checkpoint/model_checkpoint_*_{self.CFG.fold}.pt"), reverse=True)[1:]:
                        os.remove(path)
        
        return self.best_score