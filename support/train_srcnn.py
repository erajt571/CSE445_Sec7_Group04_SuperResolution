"""Training script matching the SRCNN workflow used for the project."""
from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from model import SRCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HR_DIR = PROJECT_ROOT / 'data' / 'HR_256'
LR_DIR = PROJECT_ROOT / 'data' / 'LR_x4'
CHECKPOINT = PROJECT_ROOT / 'support' / 'best_srcnn_x4.pth'
SCALE = 4
PATCH = 32
BATCH = 16
EPOCHS = 100


def make_ids(start, end):
    return [f'{i:04d}' for i in range(start, end + 1)]


def find_img(folder, image_id):
    for ext in ('.jpg', '.png', '.jpeg'):
        path = Path(folder) / f'{image_id}{ext}'
        if path.exists():
            return path
    raise FileNotFoundError(image_id)


class PatchDataset(Dataset):
    def __init__(self, ids, augment=True):
        self.ids = ids
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        lr = TF.to_tensor(Image.open(find_img(LR_DIR, image_id)).convert('RGB'))
        hr = TF.to_tensor(Image.open(find_img(HR_DIR, image_id)).convert('RGB'))
        _, h, w = lr.shape
        top = random.randint(0, h - PATCH)
        left = random.randint(0, w - PATCH)
        lr_patch = lr[:, top:top+PATCH, left:left+PATCH]
        hr_patch = hr[:, top*SCALE:(top+PATCH)*SCALE, left*SCALE:(left+PATCH)*SCALE]
        if self.augment:
            if random.random() < 0.5:
                lr_patch, hr_patch = TF.hflip(lr_patch), TF.hflip(hr_patch)
            if random.random() < 0.5:
                lr_patch, hr_patch = TF.vflip(lr_patch), TF.vflip(hr_patch)
            k = random.randint(0, 3)
            if k:
                lr_patch = torch.rot90(lr_patch, k, [1, 2])
                hr_patch = torch.rot90(hr_patch, k, [1, 2])
        return lr_patch, hr_patch


class FullImageDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        image_id = self.ids[index]
        lr = TF.to_tensor(Image.open(find_img(LR_DIR, image_id)).convert('RGB'))
        hr = TF.to_tensor(Image.open(find_img(HR_DIR, image_id)).convert('RGB'))
        return lr, hr


def bicubic_up(x):
    return F.interpolate(x, scale_factor=SCALE, mode='bicubic', align_corners=False).clamp(0, 1)


def psnr(pred, target):
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return 10 * torch.log10(1.0 / (mse + 1e-12))


def main():
    train_ids = make_ids(0, 89)
    val_ids = make_ids(90, 99)
    train_loader = DataLoader(PatchDataset(train_ids), batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(FullImageDataset(val_ids), batch_size=1, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_val = float('-inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for lr_patch, hr_patch in train_loader:
            lr_patch, hr_patch = lr_patch.to(device), hr_patch.to(device)
            lr_up = bicubic_up(lr_patch)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                sr = model(lr_up)
                loss = criterion(sr, hr_patch)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(bicubic_up(lr)).clamp(0, 1)
                vals.append(psnr(sr, hr).item())
        val_psnr = float(np.mean(vals))
        avg_loss = float(np.mean(losses))
        if val_psnr > best_val:
            best_val = val_psnr
            torch.save(model.state_dict(), CHECKPOINT)
        print(f'Epoch {epoch:03d} | MSE {avg_loss:.6f} | Val PSNR {val_psnr:.2f} dB')

    print(f'Best validation PSNR: {best_val:.2f} dB')


if __name__ == '__main__':
    main()
