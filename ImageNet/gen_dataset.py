
'''
All the following clean subsest must be extracted i.i.d. from train set (notice label balance problem)
subset1: Albumentation, 10,000
subset2: FGSM on Res-18, 10,000
subset3: ODI on (Res-18 + WRN-28), 10,000
subset4: C&W_PGD on Res-18, 10,000
subset5: MIM on Res-18, 10,000

label: soft smoothing on albumentation, sharp smoothing on adversarial examples.
'''

import argparse
import copy
import dataset_configs
# import models
import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.distributed
import ops_attack



from albumentations import (
    IAAPerspective, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine,
    Sharpen, Emboss, RandomBrightnessContrast, OneOf, Compose, Cutout, CoarseDropout, ShiftScaleRotate,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from utils import *


parser = argparse.ArgumentParser(description='generate dataset for data-centric robust learning')
parser.add_argument('--id', default='0115', type=str, help='experiment id')
parser.add_argument('--batch_size', default=256, type=int, help='')
parser.add_argument('--data', default='/data/crq/data', type=str, help='path to ImageNet dataset')
parser.add_argument('--device', default='0, 1, 2, 3, 4, 5', type=str, help='gpu device')
parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()
class Preprocessing_Layer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std

    def preprocess(self, img, mean, std):
        img = img.clone()

        img[:,0,:,:] = (img[:,0,:,:] - mean[0]) / std[0]
        img[:,1,:,:] = (img[:,1,:,:] - mean[1]) / std[1]
        img[:,2,:,:] = (img[:,2,:,:] - mean[2]) / std[2]

        return(img)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res = self.preprocess(x, self.mean, self.std)
        return res
    
class MyImageFolder(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = torchvision.datasets.folder.default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(MyImageFolder, self).__init__(root, loader, torchvision.datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


    def __getitem__(self, index: int):
        path, target = self.samples[index]
        adv_path = os.path.join(args.data, 'train_DCRTdataset', path[21:])
        sample = self.loader(path)
        adv_sample = ''
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return [sample, adv_sample], target, adv_path



def main():

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    print('device: {}'.format(device))


    traindir = os.path.join(args.data, 'train_withdiffusion')
    normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    train_dataset = MyImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, sampler=DistributedSampler(train_dataset))
    
    dataset_args = dataset_configs.__dict__['args_' + args.id]
    source_res50 = torchvision.models.resnet50(pretrained=True).eval()
    source_res50 = nn.Sequential(normalize, source_res50).to(device)
    source_incv3 = torchvision.models.inception_v3(pretrained=True).eval()
    source_incv3 = nn.Sequential(normalize, source_incv3).to(device)
    source_vgg16 = torchvision.models.vgg16(pretrained=True).eval()
    source_vgg16 = nn.Sequential(normalize, source_vgg16).to(device)
    source_model_map = {'resnet50': source_res50, 'inception_v3': source_incv3,'vgg16':source_vgg16}
    from attacks_transfer.material.models.generators import ResnetGenerator, weights_init
    gpu_id=[device.index]
    netG = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=[0])
    checkpoint='/data/crq/tempname/netG_model_epoch_10_'
    print("=> loading checkpoint '{}'".format(checkpoint))
    state_dict = torch.load(checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        # 去除module.前缀
        if k=='normalize.mean' or k=='normalize.std' or k=='module.0.mean' or k=='module.0.std':
            1
        else:  
            name = k.replace("module.1._wrapped_model.", "module.")
            new_state_dict[name] = v
    netG.load_state_dict(new_state_dict)
    print("=> loaded checkpoint '{}'".format(checkpoint))    
    from ops_attack.utils_iaa import resnet50
    resnet = resnet50()
    model_path = '/data/crq/DRL/ImageNet/models/resnet50-0676ba61.pth'
    pre_dict = torch.load(model_path)
    resnet_dict = resnet.state_dict()
    state_dict = {k:v for k,v in pre_dict.items() if k in resnet_dict.keys()}
    print("loaded pretrained weight. Len:",len(pre_dict.keys()),len(state_dict.keys()))
    resnet_dict.update(state_dict)
    model_dict = resnet.load_state_dict(resnet_dict)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess_layer = Preprocessing_Layer(mean,std)
    resnet = nn.Sequential(preprocess_layer, resnet)
    resnet.to(device)
    resnet.eval()
    i = 0
    dataset_size = len(train_loader.dataset)
    subset_size = dataset_size // 5

    subset_counters = [0] * 5

    for batch_idx, (xs, y, adv_path) in enumerate(tqdm(train_loader)):
        clean, adv = xs
        batch_size = clean.size(0)
        global_sample_count = batch_idx * batch_size
        config_idx = (global_sample_count // subset_size) % 5
        dataset_arg = dataset_args[f'subset{config_idx}']
        attacker = ops_attack.__dict__[dataset_arg['attack']]
        if attacker=='merged':
            source_args = dataset_arg['source_model']
            model_list = []
            for model_arg in source_args:
                model=source_model_map[model_arg['arch']]
                model_list.append(model)
            source_model = MergedModel(model_list=model_list, merge_mode='sampling')
        else:
            source_model = source_model_map[dataset_arg['source_model'][0]['arch']]
        from ops_attack import mi_fgsm,ni_fgsm,pi_fgsm,di_fgsm,admix,ssa,ILA,fia,NAA,GAP,GAPF,BIA,IAA,SGM,RFA
        if attacker is fia:
            resnet50_model = source_res50._modules.get('1')
            source_layers = list(enumerate(map(lambda name: (name, resnet50_model._modules.get(name)), ['conv1', 'layer1', 'layer2','layer3','layer4', 'fc'])))
            _, adv, is_adv = attacker(source_model, clean, y, source_layers[2][1][1], learning_rate=4/2550, epsilon=4/255, niters=10, dataset='imagenet')
        elif attacker is GAP or attacker is GAPF or attacker is BIA:
            _, adv, is_adv = attacker(source_model,netG,clean, y, epsilon=dataset_arg['eps'], device=device)
        elif attacker is IAA:
            _, adv, is_adv = attacker(source_model, resnet,clean, y, epsilon=dataset_arg['eps'], device=device)
        elif attacker is ILA:
            from utils import ifgsm
            ifgsm_guide = ifgsm(source_model, clean, y, learning_rate=4/2550, epsilon=4/255, niters=10, dataset='imagenet')
            source_layers=list(enumerate(map(lambda name: (name, source_model._modules.get(name)), ['conv1', 'layer1', 'layer2','layer3','layer4', 'fc'])))
            _, adv, is_adv = attacker(source_model, clean, ifgsm_guide, y, source_layers[2][1][1],learning_rate=4/2550, epsilon=4/255, niters=10, dataset='imagenet')
        elif attacker is NAA:
            source_layers=list(enumerate(map(lambda name: (name, source_model._modules.get(name)), ['conv1', 'layer1', 'layer2','layer3','layer4', 'fc'])))
            _, adv, is_adv = attacker(source_model, clean, y, source_layers[2][1][1],learning_rate=4/2550, epsilon=4/255, niters=10, dataset='imagenet')
        else:
            _, adv, is_adv = attacker(source_model, clean, y, epsilon=dataset_arg['eps'], device=device)
        if i % 100 == 99:
            print('args.local_rank: {}, attack: {}, attack success rate: {}'.format(args.local_rank, dataset_arg['attack'], is_adv.sum().item() / len(is_adv)))
        save_images(adv, adv_path)
        i += 1



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    set_random_seed(seed=2025)
    main()