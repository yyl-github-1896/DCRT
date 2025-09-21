import numpy as np
# import cv2
import os
import pdb
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torch.utils.data as Data
import torch.nn.functional as F
import dill

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

import matplotlib.pyplot as plt
import scipy.io as si
import shutil
from utils_data import *

from typing import Type, Any, Callable, Union, List, Optional
seed=59
## SGM utils
def backward_hook(gamma):
    # implement SGM through grad through ReLU
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook


def backward_hook_norm(module, grad_in, grad_out):
    # normalize the gradient to avoid gradient explosion or vanish
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


def register_hook_for_resnet(model, arch, gamma):
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if arch in ['resnet50', 'resnet101', 'resnet152']:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        if 'relu' in name and not '0.relu' in name:
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        # if len(name.split('.')) == 3:
        if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
            module.register_backward_hook(backward_hook_norm)


def register_hook_for_densenet(model, arch, gamma):
    # There are 2 ReLU in Conv module of DenseNet-121/169/201.
    gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)
    for name, module in model.named_modules():
        if 'relu' in name and not 'transition' in name:
            module.register_backward_hook(backward_hook_sgm)


def sow_images(images):
    """Sow batch of torch images (Bx3xWxH) into a grid PIL image (BWxHx3)

    Args:
        images: batch of torch images.

    Returns:
        The grid of images, as a numpy array in PIL format.
    """
    images = torchvision.utils.make_grid(
        images
    )  # sow our batch of images e.g. (4x3x32x32) into a grid e.g. (3x32x128)
    
    mean_arr, stddev_arr = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # denormalize
    for c in range(3):
        images[c, :, :] = images[c, :, :] * stddev_arr[c] + mean_arr[c]

    images = images.cpu().numpy()  # go from Tensor to numpy array
    # switch channel order back from
    # torch Tensor to PIL image: going from 3x32x128 - to 32x128x3
    images = np.transpose(images, (1, 2, 0))
    return images
class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean).cuda()
        std = torch.as_tensor(std).cuda()
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
invNormalize = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
Normlize_Trans = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def update_adv(X, X_pert, pert, epsilon):
    X = X.clone().detach()
    X_pert = X_pert.clone().detach()
    X_pert = invNormalize(X_pert)
    X = invNormalize(X)
    X_pert = X_pert + pert
    noise = (X_pert - X).clamp(-epsilon, epsilon)
    X_pert = X + noise
    X_pert = X_pert.clamp(0, 1)
    X_pert = Normlize_Trans(X_pert)
    return X_pert.clone().detach()

def ifgsm(
    model,
    X,
    y,
    niters=10,
    epsilon=0.03,
    visualize=False,
    learning_rate=0.005,
    display=4,
    defense_model=False,
    setting="regular",
    dataset="cifar10",
    use_Inc_model = False,
):
    """Perform ifgsm attack with respect to model on images X with labels y

    Args:
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        y: labels corresponding to the batch of images
        niters: number of iterations of ifgsm to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        visualize: whether you want to visualize the perturbations or not
        learning_rate: learning rate of ifgsm
        display: number of images to display in visualization
        defense_model: set to true if you are using a defended model,
        e.g. ResNet18Defended, instead of the usual ResNet18
        setting: 'regular' is usual ifgsm, 'll' is least-likely ifgsm, and
        'nonleaking' is non-label-leaking ifgsm
        dataset: dataset the images are from, 'cifar10' | 'imagenet'

    Returns:
        The batch of adversarial examples corresponding to the original images
    """
    model.eval()
    out = None
    if defense_model:
        out = model(X)[0]
    else:
        out = model(X)
    y_ll = out.min(1)[1]  # least likely model output
    y_ml = out.max(1)[1]  # model label

    X_pert = X.clone()
    X_pert.requires_grad = True
    for i in range(niters):
        output_perturbed = None
        if defense_model:
            output_perturbed = model(X_pert)[0]
        else:
            output_perturbed = model(X_pert)

        y_used = y
        ll_factor = 1
        if setting == "ll":
            y_used = y_ll
            ll_factor = -1
        elif setting == "noleaking":
            y_used = y_ml

        loss = nn.CrossEntropyLoss()(output_perturbed, y_used)
        loss.backward()
        pert = ll_factor * learning_rate * X_pert.grad.detach().sign()

        # perform visualization
        if visualize is True and i == niters - 1:
            np_image = sow_images(X[:display].detach())
            np_delta = sow_images(pert[:display].detach())
            np_recons = sow_images(
                (X_pert.detach() + pert.detach()).clamp(-1, 1)[:display]
            )

            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(3, 1, 1)
            plt.axis("off")
            plt.imshow(np_recons)
            fig.add_subplot(3, 1, 2)
            plt.axis("off")
            plt.imshow(np_image)
            fig.add_subplot(3, 1, 3)
            plt.axis("off")
            plt.imshow(np_delta)
            plt.show()
        # end visualization

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

    return X_pert.clone().detach()
class Proj_Loss(torch.nn.Module):
    def __init__(self):
        super(Proj_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)
        x_norm = x / x.norm()

        proj_loss = torch.mm(y, x_norm.transpose(0, 1)) / x.norm()
        return proj_loss
class Mid_layer_target_Loss(torch.nn.Module):
    def __init__(self):
        super(Mid_layer_target_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)

        x_norm = x / x.norm()
        if (y == 0).all():
            y_norm = y
        else:
            y_norm = y / y.norm()
        angle_loss = torch.mm(x_norm, y_norm.transpose(0, 1))
        magnitude_gain = y.norm() / x.norm()
        return angle_loss + magnitude_gain * coeff
def ILA(
    model,
    X,
    X_attack,
    y,
    feature_layer,
    niters=10,
    epsilon=0.01,
    coeff=1.0,
    learning_rate=1,
    dataset="cifar10",
    use_Inc_model = False,
    with_projection=True,
):
    """Perform ILA attack with respect to model on images X with labels y

    Args:
        with_projection: boolean, specifies whether projection should happen
        in the attack
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        X_attack: starting adversarial examples of ILA that will be modified
        to become more transferable
        y: labels corresponding to the batch of images
        feature_layer: layer of model to project on in ILA attack
        niters: number of iterations of the attack to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        coeff: coefficient of magnitude loss in ILA attack
        visualize: whether you want to visualize the perturbations or not
        learning_rate: learning rate of the attack
        dataset: dataset the images are from, 'cifar10' | 'imagenet'

    Returns:
        The batch of modified adversarial examples, examples have been
        augmented from X_attack to become more transferable
    """
    X = X.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o

    h = feature_layer.register_forward_hook(get_mid_output)

    out = model(X)
    mid_original = torch.zeros(mid_output.size()).cuda()
    mid_original.copy_(mid_output)

    out = model(X_attack)
    mid_attack_original = torch.zeros(mid_output.size()).cuda()
    mid_attack_original.copy_(mid_output)

    adversaries = []
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    for iter_n in range(niters):
        output_perturbed = model(X_pert)

        # generate adversarial example by max middle layer pertubation
        # in the direction of increasing loss
        if with_projection:
            loss = Proj_Loss()(
                mid_attack_original.detach(), mid_output, mid_original.detach(), coeff
            )
        else:
            loss = Mid_layer_target_Loss()(
                mid_attack_original.detach(), mid_output, mid_original.detach(), coeff
            )

        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        
        
        if (iter_n+1) % 10 == 0:
            adversaries.append(X_pert.clone().detach())

    h.remove()
    return X_pert.clone().detach()
def get_source_layers(model_name, model):
    if model_name == 'ResNet18':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)), ['conv1', 'bn1', 'layer1', 'layer2','layer3','layer4','fc'])))
    
    elif model_name == 'ResNet50':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)), ['conv1', 'layer1', 'layer2','layer3','layer4', 'fc'])))
    
    elif model_name == 'DenseNet121':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get('features')._modules.get(name)), ['conv0', 'denseblock1', 'transition1', 'denseblock2', 'transition2', 'denseblock3', 'transition3', 'denseblock4', 'norm5']))
        layer_list.append(('classifier', model._modules.get('classifier')))
        return list(enumerate(layer_list))
                                             
    elif model_name == 'SqueezeNet1.0':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: ('layer '+name, model._modules.get('features')._modules.get(name)), ['0','3','4','5','7','8','9','10','12']))
        layer_list.append(('classifier', model._modules.get('classifier')._modules.get('1')))
        return list(enumerate(layer_list))
    
    elif model_name == 'alexnet':
        # exclude avgpool
        layer_list = list(map(lambda name: ('layer '+name, model._modules.get('features')._modules.get(name)), ['0','3','6','8','10']))
        layer_list += list(map(lambda name: ('layer '+name, model._modules.get('classifier')._modules.get(name)), ['1','4','6']))
        return list(enumerate(layer_list))
    
    elif model_name == 'IncRes-v2':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)), ['conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'maxpool_3a', 'conv2d_3b', 'conv2d_4a', 'maxpool_5a', 'mixed_5b', 'repeat', 'mixed_6a','repeat_1', 'mixed_7a', 'repeat_2', 'block8', 'conv2d_7b', 'avgpool_1a', 'last_linear'])))

    elif model_name == 'Inc-v4':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get('features')._modules.get(name)), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']))
        return list(enumerate(layer_list))
                                             
    elif model_name == 'Inc-v3':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get(name)), ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c']))
        return list(enumerate(layer_list))
    
    else:
        # model is not supported
        assert False
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.utils
from torchvision import models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights,Inception_V3_Weights,DenseNet121_Weights,VGG19_BN_Weights
from utils_data import *
model_tar_1 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,transform_input=True).eval()
model = torchvision.models.resnet50(pretrained=True).cuda().eval()
# from transformers import AutoModelForImageClassification, AutoTokenizer
# model = AutoModelForImageClassification.from_pretrained("/data/crq/DRL/ImageNet/ckpt/defense/microsoft")
model_tar_2 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval()
model_tar_3 = models.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1).eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]   
preprocess_layer = Preprocessing_Layer(mean,std)

# model = nn.Sequential(preprocess_layer, model).eval()
model_tar_1 = nn.Sequential(preprocess_layer, model_tar_1).eval()
model_tar_2 = nn.Sequential(preprocess_layer, model_tar_2).eval()
model_tar_3 = nn.Sequential(preprocess_layer, model_tar_3).eval()

# model.eval()
model.to(device)
model_tar_1.to(device)
model_tar_2.to(device)
model_tar_3.to(device)

transform_299 = transforms.Compose([
transforms.Resize(299),
transforms.CenterCrop(299),
transforms.ToTensor(),
])


transform_224 = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
])   
 
# image_size = 299 for inception_v3 and image_size = 224 for resnet50, densenet121, and vgg16_bn  
image_size = 224
 
if image_size ==299:
    transform = transform_299
else:
    transform = transform_224
img_dir='/data/crq/DRL/ImageNet/kmeans/img'
label_dir = '/data/crq/DRL/ImageNet/kmeans/km.csv'
# img_dir = '/data/crq/data/val_clean'
# label_dir = '/data/crq/data/val_rs.csv'
val_set = MyDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        transform=transform,
        png=False,
    )
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False,
                                              num_workers=4)
# val_json = '/data/crq/DRL/val10000.json'
# val_loader = torch.utils.data.DataLoader(ImagePathDataset.from_path(config_path = val_json,transform=transform,return_paths=True),batch_size=50, shuffle=False,num_workers=1, pin_memory=True)
# val_loader = torch.utils.data.DataLoader(MyDataset(img_dir='/data/crq/data/val',
#         label_dir='/data/crq/data/val_rs.csv',
#         transform=transform,
#         png=False,),batch_size=50, shuffle=False,num_workers=1, pin_memory=True)
epsilon = 4.0 / 255.0
num_iteration=10
step_size = epsilon / num_iteration
# step_size = 2.0 / 255.0
check_point = 5

# ILA
save_dir = os.path.join('/data/crq/kmean','ILA',str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
suc = np.zeros((3,num_iteration // check_point))

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    ifgsm_guide = ifgsm(model, img, labels, learning_rate=step_size, epsilon=4/255, niters=10, dataset='imagenet')
    # fgsm_guide = fgsm(model, image_batch, label_batch, epsilon=opt.epsilon, dataset='imagenet')
    source_layers = get_source_layers(model_name='ResNet50', model=model)
    adversaries = ILA(model, img, ifgsm_guide, labels, source_layers[2][1][1], learning_rate=step_size, epsilon=4/255, niters=10, dataset='imagenet')
    adversaries = torch.clamp(adversaries, min=0, max=1)
    save_images(adversaries.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)


# FIA
def FIA(
    model,
    X,    
    y,    
    feature_layer,
    N=30,
    drop_rate=0.3,
    niters=10,    
    epsilon=0.01,
    learning_rate=1,
    decay=1,
    dataset="cifar10",
    use_Inc_model = False,    
):
    """
        Feature Importance-aware Attack
    """
    X = X.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o        
    
    def get_mid_grad(m, i, o):
        global mid_grad
        mid_grad = o        

    h = feature_layer.register_forward_hook(get_mid_output)
    h2 = feature_layer.register_full_backward_hook(get_mid_grad)

    '''
        Set Seeds
    '''
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    '''
        Gradient Aggregate
    '''
    agg_grad = 0
    for _ in range(N):        
        X_random = torch.zeros(X.size()).cuda()
        X_random.copy_(X).detach()
        X_random.requires_grad = True
        Mask = torch.bernoulli(torch.ones_like(X_random)*(1-drop_rate))
        X_random = X_random * Mask
        output_random = model(X_random)        
        loss = 0
        for batch_i in range(X.shape[0]):
            loss += output_random[batch_i][y[batch_i]]        
        model.zero_grad()
        loss.backward()        
        agg_grad += mid_grad[0].detach()    
    for batch_i in range(X.shape[0]):
        agg_grad[batch_i] /= agg_grad[batch_i].norm(2)
    h2.remove()   

    adversaries = []

    momentum = 0
    for iter_n in range(niters):
        output_perturbed = model(X_pert)
        loss = (mid_output * agg_grad).sum()
        model.zero_grad()
        loss.backward()

        # momentum = decay * momentum + X_pert.grad / torch.sum(torch.abs(X_pert.grad))
        # pert = -learning_rate * momentum.sign()

        # No Momentum
        pert = -learning_rate * X_pert.grad.detach().sign()

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

        if (iter_n+1) % 10 == 0:
            adversaries.append(X_pert.clone().detach())

    h.remove()        
    # return adversaries    
    return X_pert.clone().detach()
def NAA(
    model,
    X,    
    y,    
    feature_layer,
    N=30,    
    niters=10,    
    epsilon=0.01,
    learning_rate=1,
    decay=1,        
    dataset="cifar10",
    use_Inc_model = False,    
):
    """
        NAA attack with default Linear Transformation Functions and Weighted Attribution gamma = 1
    """
    X = X.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o        
    
    def get_mid_grad(m, i, o):
        global mid_grad
        mid_grad = o        

    h = feature_layer.register_forward_hook(get_mid_output)
    h2 = feature_layer.register_full_backward_hook(get_mid_grad)

    '''
        Set Seeds
    '''
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    '''
        Gradient Aggregate
    '''
    agg_grad = 0
    for iter_n in range(N): 
        X_Step = torch.zeros(X.size()).cuda()
        X_Step = X_Step + X.clone().detach() * iter_n / N
        
        output_random = model(X_Step)        
        output_random  = torch.softmax(output_random, 1)

        loss = 0
        for batch_i in range(X.shape[0]):
            loss += output_random[batch_i][y[batch_i]]        
        model.zero_grad()
        loss.backward()        
        agg_grad += mid_grad[0].detach()
    agg_grad /= N    
    h2.remove()   

    adversaries = []

    X_prime = torch.zeros(X.size()).cuda()
    model(X_prime)
    Output_prime = mid_output.detach().clone()

    momentum = 0
    for iter_n in range(niters):
        output_perturbed = model(X_pert)
        loss = ((mid_output - Output_prime) * agg_grad).sum()
        model.zero_grad()
        loss.backward()

        # momentum = decay * momentum + X_pert.grad / torch.sum(torch.abs(X_pert.grad))
        # pert = -learning_rate * momentum.sign()

        # No Momentum
        pert = -learning_rate * X_pert.grad.detach().sign()

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

        if (iter_n+1) % 10 == 0:
            adversaries.append(X_pert.clone().detach())

    h.remove()        
    # return adversaries    
    return X_pert.clone().detach()
# FIA
save_dir = os.path.join('/data/crq/kmean','FIA',str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
suc = np.zeros((3,num_iteration // check_point))

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    img = images.clone()
    # fgsm_guide = fgsm(model, image_batch, label_batch, epsilon=opt.epsilon, dataset='imagenet')
    source_layers = get_source_layers(model_name='ResNet50', model=model)
    adversaries = FIA(model, img, labels, source_layers[2][1][1], learning_rate=step_size, epsilon=4/255, niters=10, dataset='imagenet')
    adversaries = torch.clamp(adversaries, min=0, max=1)
    save_images(adversaries.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)

# NAA
save_dir = os.path.join('/data/crq/kmean','NAA',str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
suc = np.zeros((3,num_iteration // check_point))

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    img = images.clone()
    # fgsm_guide = fgsm(model, image_batch, label_batch, epsilon=opt.epsilon, dataset='imagenet')
    source_layers = get_source_layers(model_name='ResNet50', model=model)
    adversaries = NAA(model, img, labels, source_layers[2][1][1], learning_rate=step_size, epsilon=4/255, niters=10, dataset='imagenet')
    adversaries = torch.clamp(adversaries, min=0, max=1)
    save_images(adversaries.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)