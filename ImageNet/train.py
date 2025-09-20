from __future__ import print_function

import argparse
import os
import random
import shutil
from tqdm import tqdm
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
# from config import *
from torchvision import datasets, transforms, models
import albumentations as A
CORRUPTIONS = [
    'shot_noise', 'impulse_noise', 
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 
    'brightness', 'contrast',
     'pixelate', 
]
def add_impulse_noise(image, **kwargs):
    """对输入图像添加 impulse noise (椒盐噪声)"""
    noise_ratio = 0.02 
    output = image.copy()
    num_salt = np.ceil(noise_ratio * image.size * 0.5)
    num_pepper = np.ceil(noise_ratio * image.size * 0.5)
    coords_salt = tuple(
        np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]
    )
    coords_pepper = tuple(
        np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]
    )
    output[coords_salt] = 255  
    output[coords_pepper] = 0   

    return output 
def get_corruption_transform(corruption_type):
    """根据 corruption_type 返回对应的 albumentations 增强操作"""
    if corruption_type == 'shot_noise':
        return A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0)
    elif corruption_type == 'impulse_noise':
        return A.Lambda(image=add_impulse_noise, p=1.0)
    elif corruption_type == 'glass_blur':
        return A.GlassBlur(sigma=0.3, max_delta=1, iterations=1, p=1.0)
    elif corruption_type == 'motion_blur':
        return A.MotionBlur(blur_limit=(2, 4), p=1.0)
    elif corruption_type == 'zoom_blur':
        return A.ZoomBlur(max_factor=1.2, p=1.0)
    elif corruption_type == 'snow':
        return A.RandomSnow(brightness_coeff=1.1, p=1.0)
    elif corruption_type == 'brightness':
        return A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)
    elif corruption_type == 'contrast':
        return A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.15, p=1.0)
    elif corruption_type == 'pixelate':
        return A.PixelDropout(dropout_prob=0.1, p=1.0)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

parser = argparse.ArgumentParser(description='train source models or robust models')
parser.add_argument('--id', default='dcrt', type=str, help='experiment id')
parser.add_argument('-ar', '--ar', dest='ar', action='store_true', help='use alignment regularization')
parser.add_argument('-ds', '--ds', dest='ds', action='store_true', help='use data selection')
parser.add_argument('--arch', default='resnet50', type=str, help='model architecture')
parser.add_argument('--pretrain', type=str, help='path to pretrained checkpoint')
parser.add_argument('--seed', default=2023, type=int, help='random seed')
parser.add_argument('--device', default='0', type=str, help='gpu device')

args = parser.parse_args()

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)
class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)
def create_imagenet_subset_loader(data_path, original_class_counts, transform, batch_size, num_workers):
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision.datasets import ImageFolder
    full_dataset = ImageFolder(root=data_path, transform=transform)
    extended_class_indices = {}
    for idx, (path, label) in enumerate(full_dataset.samples):
        if label not in extended_class_indices:
            extended_class_indices[label] = []
        extended_class_indices[label].append(idx)
    selected_indices = []
    for label, indices in extended_class_indices.items():
        target_count = original_class_counts.get(label, 0)
        if target_count == 0:
            continue
        n_samples = min(target_count, len(indices))
        selected = np.random.choice(indices, n_samples, replace=False)
        selected_indices.extend(selected.tolist())
    subset_dataset = Subset(full_dataset, selected_indices)
    loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
def main():
    root_data_path = '/data/crq/data/train_DCRTdataset'  # 新的混合数据集路径
    root_ckpt_path = f'./checkpoints/{args.id}'  # 检查点保存路径
    
    print(f'Dataset path: {root_data_path}')
    print(f'Checkpoint path: {root_ckpt_path}')

    # 模型架构配置
    arch_configs = {
        'resnet50': {
            'epochs': 50,  # 修改2：训练周期改为30
            'batch_size': 256,
            'optimizer': 'SGD',
            'optim_hparams': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
            'scheduler': 'CosineAnnealingLR',
            'scheduler_hparams': {'T_max': 30},  # 修改3：调整学习率衰减点
            'ar_weight': 0
        }
    }

    # load model
    model= torchvision.models.resnet50(pretrained=True)
    model = model.cuda()
    best_acc = 0
    train_args = arch_configs[args.arch]
    optimizer = optim.__dict__[train_args['optimizer']](
        model.parameters(), **train_args['optim_hparams'])
    scheduler = torch.optim.lr_scheduler.__dict__[train_args['scheduler']](
        optimizer, **train_args['scheduler_hparams'])
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    original_imagenet_path = "/data/crq/data/train"
    from torchvision.datasets import ImageFolder
    original_dataset = ImageFolder(root=original_imagenet_path)
    original_class_counts = {}
    for _, label in original_dataset.samples:
        original_class_counts[label] = original_class_counts.get(label, 0) + 1
    for epoch in tqdm(range(train_args['epochs'])):
        train_loader = create_imagenet_subset_loader(
        data_path=root_data_path,
        original_class_counts=original_class_counts,  # 使用前面获取的类别数量字典
        transform=transform_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

        # 训练过程
        train_loss, train_acc,_ = train(
            train_loader, model, optimizer, 
            ar=args.ar, 
            ar_weight=train_args['ar_weight']
        )

        # 更新学习率
        scheduler.step()

        # 保存检查点
        best_acc = max(train_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': train_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, args.arch, root_ckpt_path)

        print(f'Epoch {epoch+1}/{train_args["epochs"]} | Acc: {train_acc:.2f}% | Best: {best_acc:.2f}%')

def train(train_loader, model, optimizer, ar=False, ar_weight=0):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        outputs = model(inputs)
        total_loss = F.cross_entropy(outputs, targets)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        train_loss += total_loss.item()

    avg_loss = train_loss / len(train_loader)
    acc = 100. * correct / total
    
    return avg_loss, acc, None

def save_checkpoint(state, arch, root_path):
    os.makedirs(root_path, exist_ok=True)
    torch.save(state, os.path.join(root_path, f'{arch}_best.pth.tar'))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()