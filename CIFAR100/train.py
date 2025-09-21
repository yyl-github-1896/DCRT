from __future__ import print_function

import argparse
import os
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
# from config import *
from utils import load_model, AverageMeter, accuracy, TrainDataset, set_random_seed
import albumentations as A
CORRUPTIONS = [
    # 'gaussian_noise', 
    'shot_noise', 'impulse_noise', 
    # 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 
    # 'frost', 'fog',
    'brightness', 'contrast',
    #  'elastic_transform', 
     'pixelate', 
    #  'ImageCompression'
]
def add_impulse_noise(image, **kwargs):
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

set_random_seed(2023)

parser = argparse.ArgumentParser(description='train source models or robust models')
parser.add_argument('--id', default='dcrtgmonly', type=str, help='experiment id')
parser.add_argument('-ar', '--ar', dest='ar', action='store_true', help='use alignment regularization')
parser.add_argument('-ds', '--ds', dest='ds', action='store_true', help='use data selection')
parser.add_argument('--arch', default='preactresnet18', type=str, help='model architecture')
parser.add_argument('--pretrain', default='/data/crq/DRL/CIFAR10/DCRT/checkpoints/clean/preactresnet18.pth.tar', type=str, help='path to pretrained checkpoint')
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
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)
def main():
    root_data_path = '/data/crq/DRL/CIFAR10/DCRT/datasets/dcrt' 
    root_ckpt_path = f'./checkpoints/{args.id}'
    
    print(f'Dataset path: {root_data_path}')
    print(f'Checkpoint path: {root_ckpt_path}')

    arch_configs = {
        'preactresnet18': {
            'epochs': 30, 
            'batch_size': 256,
            'optimizer': 'SGD',
            'optim_hparams': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
            'scheduler': 'CosineAnnealingLR',
            'scheduler_hparams': {'T_max': 30},  
            'ar_weight': 0
        },
    }

    model = load_model(args.arch)
    if os.path.exists(args.pretrain):
        if args.arch == 'resnet20':
            state_dict=torch.load(args.pretrain, map_location='cpu')["model0"]
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                if new_key.startswith("1."):
                    new_key = new_key[2:]  
                elif new_key.startswith("0."):
                    continue  
                elif new_key.startswith("normalizer."):
                    continue 
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(torch.load(args.pretrain)["state_dict"])
    model = model.cuda()
    best_acc = 0

    train_args = arch_configs[args.arch]
    optimizer = optim.__dict__[train_args['optimizer']](
        model.parameters(), **train_args['optim_hparams'])

    scheduler = torch.optim.lr_scheduler.__dict__[train_args['scheduler']](
        optimizer, **train_args['scheduler_hparams'])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    confidence_path = os.path.join(root_data_path, 'confidence.npy')
    np.save(confidence_path, np.zeros(50000))  

    for epoch in tqdm(range(train_args['epochs'])):
        trainset = TrainDataset(
            transform=transform_train,
            img_path=os.path.join(root_data_path, 'images.npy'),
            label_path=os.path.join(root_data_path, 'labels.npy'),
            confidence_path=confidence_path
        )
        
        trainloader = data.DataLoader(
            trainset, 
            batch_size=train_args['batch_size'], 
            shuffle=True, 
            num_workers=4
        )

        train_loss, train_acc,_ = train(
            trainloader, model, optimizer, 
            ar=args.ar, 
            ar_weight=train_args['ar_weight'],
            confidence=np.load(confidence_path) if args.ds else None
        )

        scheduler.step()

        best_acc = max(train_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': train_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, args.arch, root_ckpt_path)

        print(f'Epoch {epoch+1}/{train_args["epochs"]} | Acc: {train_acc:.2f}% | Best: {best_acc:.2f}%')

def train(trainloader, model, optimizer, ar, ar_weight, confidence):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
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

    avg_loss = train_loss / len(trainloader)
    acc = 100. * correct / total
    
    return avg_loss, acc, None

def save_checkpoint(state, arch, root_path):
    os.makedirs(root_path, exist_ok=True)
    torch.save(state, os.path.join(root_path, f'{arch}_best.pth.tar'))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()
