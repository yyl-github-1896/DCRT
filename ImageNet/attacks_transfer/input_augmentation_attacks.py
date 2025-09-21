import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import scipy.stats as st
from torchvision.models import ResNet50_Weights,Inception_V3_Weights,DenseNet121_Weights,VGG19_BN_Weights,VGG16_Weights
from utils_data import *

seed=59
##define TI
def gkern(kernlen=5, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def TI(grad_in, kernel_size=5):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()    
    grad_out = F.conv2d(grad_in, gaussian_kernel, bias=None, stride=1, padding=((kernel_size-1)/2,(kernel_size-1)/2), groups=3) #TI
    return grad_out


def TI_multi(X_in, in_size, kernel_size):
    a = (kernel_size-1)/2/in_size
    X_out = transforms.RandomAffine(0,translate=(a,a))(X_in)
    return X_out

##define DI
def DI(X_in, in_size, out_size):
    rnd = np.random.randint(in_size, out_size,size=1)[0]
    h_rem = out_size - rnd
    w_rem = out_size - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)), (pad_left,pad_top,pad_right,pad_bottom), mode='constant', value=0)
        return  X_out 
    else:
        return  X_in
    
    

#resnet50 inception_v3 densenet121 vgg16_bn
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

model_tar_1 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,transform_input=True).eval()
# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
# Use a pipeline as a high-level helper
# from tensorflow.keras.applications import ResNet50

# # 创建 ResNet50 模型
# model = ResNet50()  # 不加载预训练权重

# # 加载自定义权重
# model.load_weights('/data/crq/DRL/ImageNet/ckpt/defense/transformer/tf_model.h5')
# 加载模型
# from transformers import AutoModelForImageClassification, AutoTokenizer
# model = AutoModelForImageClassification.from_pretrained("/data/crq/DRL/ImageNet/ckpt/defense/microsoft")
# tokenizer = AutoTokenizer.from_pretrained("/data/crq/DRL/ImageNet/ckpt/defense/microsoft")

model_tar_2 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval()
model_tar_3 = models.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1).eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]   
preprocess_layer = Preprocessing_Layer(mean,std)

model = nn.Sequential(preprocess_layer, model).eval()
model_tar_1 = nn.Sequential(preprocess_layer, model_tar_1).eval()
model_tar_2 = nn.Sequential(preprocess_layer, model_tar_2).eval()
model_tar_3 = nn.Sequential(preprocess_layer, model_tar_3).eval()

for param in model.parameters():
    param.requires_grad = False  
for param in model_tar_1.parameters():
    param.requires_grad = False  
for param in model_tar_2.parameters():
    param.requires_grad = False  
for param in model_tar_3.parameters():
    param.requires_grad = False  

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
#         label_dir='/data/crq/data/val_rs.csv',
#         transform=transform,
#         png=False,),batch_size=50, shuffle=False,num_workers=1, pin_memory=True)
epsilon = 4.0 / 255.0
num_iteration = 10
step_size = epsilon / num_iteration 
check_point = 5
multi_copies = 5


# DIS
# save_dir = os.path.join('/data/crq/out','DI','vgg16')
# save_dir = os.path.join('/data/crq/out','DI','res50')
save_dir = os.path.join('/data/crq/kmean','DI',str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
suc = np.zeros((3,num_iteration // check_point))

# for i in range(1):
#     ((images, labels), path)= next(iter(val_loader))
for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    for j in range(num_iteration):
        img_x = img + img.new_zeros(img.size())
        img_x.requires_grad_(True)
        if not multi_copies:
            logits = model(DI(img_x,299,330))
            loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
            loss.backward()
            input_grad = img_x.grad.clone()
        else:               
            input_grad = 0
            for c in range(multi_copies):
                logits = model(DI(img_x,299,330))
                loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
                loss.backward()
                input_grad = input_grad + img_x.grad.clone()     
                img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()
        if j == 9: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)
print('DI success rate:')
print(suc/1000)


# # TI
# # save_dir = os.path.join('/data/crq/out','TI','vgg16')
# # save_dir = os.path.join('/data/crq/out','TI','res50')
# save_dir = os.path.join('/data/crq/out','TI','res50eps8')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# suc = np.zeros((3,num_iteration // check_point))

# for i, ((images, labels), path) in enumerate(val_loader):
#     images = images.to(device)
#     labels = labels.to(device)
#     img = images.clone()
#     for j in range(num_iteration):
#         img_x = img + img.new_zeros(img.size())
#         img_x.requires_grad_(True)
#         if not multi_copies:
#             logits = model(TI(img_x))
#             loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
#             loss.backward()
#             input_grad = img_x.grad.clone()
#             input_grad = TI(input_grad)
#         else:               
#             input_grad = 0
#             for c in range(multi_copies):
#                 logits = model(TI_multi(img_x,image_size,5))
# #                 TI_multi(X_in, in_size, kernel_size)
#                 loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
#                 loss.backward()
#                 input_grad = input_grad + img_x.grad.clone()     
#                 img_x.grad.zero_()
#         img = img.data + step_size * torch.sign(input_grad)
#         img = torch.where(img > images + epsilon, images + epsilon, img)
#         img = torch.where(img < images - epsilon, images - epsilon, img)
#         img = torch.clamp(img, min=0, max=1)

#         flag = (j+1) % check_point
#         if flag == 0:
#             point = j // check_point
#             suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
#             suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
#             suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()                        
#         if j == 49: 
#             save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)              
# print('TI success rate:')
# print(suc/10000)



# # Admix
# save_dir = os.path.join('/data/crq/out','Admix','vgg16')
# save_dir = os.path.join('/data/crq/out','Admix','res50')
save_dir = os.path.join('/data/crq/kmean','Admix',str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
suc = np.zeros((3,num_iteration // check_point))

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True) 
        input_grad = 0
        for c in range(multi_copies):
            #For a dataset of 5000 images with 5 images per class, this simple random selection will almost always (4995/5000=99.9%) yield an image from a different class.
            img_other = img[torch.randperm(img.shape[0])].view(img.size())
            logits = model(img_x + 0.2 * img_other)
            loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
            loss.backward()
            input_grad = input_grad + img_x.grad.clone()              
            img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)

        flag = (j+1) % check_point
        if flag == 0:
            point = j // check_point
            suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()

        if j == 9: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)                       
print('Admix success rate:')
print(suc/1000)


# # # SI
# # save_dir = os.path.join('/data/crq/out','SI','microres50')
# # if not os.path.exists(save_dir):
# #     os.makedirs(save_dir)
# # suc = np.zeros((3,num_iteration // check_point))

# # for i, ((images, labels), path) in enumerate(val_loader):
# #     images = images.to(device)
# #     labels = labels.to(device)
# #     img = images.clone()
# #     for j in range(num_iteration):
# #         img_x = img + img.new_zeros(img.size())
# #         img_x.requires_grad_(True)
# #         if not multi_copies:
# #             logits = model(img_x * random.randint(5, 10)/10)
# #             loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
# #             loss.backward()
# #             input_grad = img_x.grad.clone()
# #         else:               
# #             input_grad = 0
# # #             for i in range(multi_copies):
# #             for c in [1,2,4,8,16]:
# #                 logits = model(img_x / c)
# #                 loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
# #                 loss.backward()
# #                 input_grad = input_grad + img_x.grad.clone()     
# #                 img_x.grad.zero_()
# #         img = img.data + step_size * torch.sign(input_grad)
# #         img = torch.where(img > images + epsilon, images + epsilon, img)
# #         img = torch.where(img < images - epsilon, images - epsilon, img)
# #         img = torch.clamp(img, min=0, max=1)

# #         flag = (j+1) % check_point
# #         if flag == 0:
# #             point = j // check_point
# #             suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
# #             suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
# #             suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()
# #         if j == 49: 
# #             save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)
# # print('SI success rate:')
# # print(suc/1000)


# # # VT
# # save_dir = os.path.join('/data/crq/out','VT','microres50')
# # if not os.path.exists(save_dir):
# #     os.makedirs(save_dir)
# # suc = np.zeros((3,num_iteration // check_point)) 

# # for i, ((images, labels), path) in enumerate(val_loader):
# #     images = images.to(device)
# #     labels = labels.to(device)
# #     img = images.clone()
# #     variance = 0
# #     for j in range(num_iteration):
# #         img_x = img
# #         img_x.requires_grad_(True)
# #         logits = model(img_x)
# #         loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
# #         loss.backward()
# #         new_grad = img_x.grad.clone()    
        
# #         global_grad = 0
# #         for c in range(multi_copies):
# #             logits = model(img_x + img.new(img.size()).uniform_(-1.5 * epsilon,1.5 * epsilon))
# #             loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
# #             loss.backward()
# #             global_grad = global_grad + img_x.grad.clone()  
# #             img_x.grad.zero_()            
# #         input_grad = new_grad + variance 
# #         variance = global_grad / multi_copies - new_grad
# #         img = img.data + step_size * torch.sign(input_grad)
# #         img = torch.where(img > images + epsilon, images + epsilon, img)
# #         img = torch.where(img < images - epsilon, images - epsilon, img)
# #         img = torch.clamp(img, min=0, max=1)
    
# #         flag = (j+1) % check_point
# #         if flag == 0:
# #             point = j // check_point
# #             suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
# #             suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
# #             suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()
# #         if j == 49: 
# #             save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)             
# # print('VT success rate:')
# # print(suc/1000)

# DI-TI-SI
# def gkern(kernlen=21, nsig=3):
#     """Returns a 2D Gaussian kernel array."""
#     x = np.linspace(-nsig, nsig, kernlen)
#     kern1d = st.norm.pdf(x)
#     kernel_raw = np.outer(kern1d, kern1d)
#     kernel = kernel_raw / kernel_raw.sum()
#     return kernel
# def input_diversity(input_tensor):
#     """apply input transformation to enhance transferability: padding and resizing (DIM)"""
#     image_width = 224
#     image_height = 224
#     image_resize = 331
#     # for vit
#     prob = 0.5        # probability of using diverse inputs
#     rnd = torch.randint(image_width, image_resize, ())   # uniform distribution
#     rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='nearest')
#     h_rem = image_resize - rnd
#     w_rem = image_resize - rnd
#     pad_top = torch.randint(0, h_rem, ())
#     pad_bottom = h_rem - pad_top
#     pad_left = torch.randint(0, w_rem, ())
#     pad_right = w_rem - pad_left
#     # pad的参数顺序在pytorch里面是左右上下，在tensorflow里是上下左右，而且要注意pytorch的图像格式是BCHW, tensorflow是CHWB
#     padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0))
#     if torch.rand(1) < prob:
#         ret = padded
#     else:
#         ret = input_tensor
#     ret = F.interpolate(ret, [image_height, image_width], mode='nearest')
#     return ret
# kernel = gkern(7, 3).astype(np.float32)
# # 要注意Pytorch是BCHW, tensorflow是BHWC
# stack_kernel = np.stack([kernel, kernel, kernel])
# stack_kernel = np.expand_dims(stack_kernel, 1)  # batch, channel, height, width = 3, 1, 7, 7
# stack_kernel = torch.tensor(stack_kernel).cuda()

# # save_dir = os.path.join('/data/crq/out','DI-TI-SI','vgg16')
# # save_dir = os.path.join('/data/crq/out','DI-TI-SI','res50')
# save_dir = os.path.join('/data/crq/out','DI-TI-SI','res50eps8')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# suc = np.zeros((3,num_iteration // check_point))
# for i, ((images, labels), path) in enumerate(val_loader):
#     images = images.to(device)
#     labels = labels.to(device)
#     img = images.clone()
#     img.requires_grad = True
#     y_batch = torch.cat((labels, labels, labels, labels, labels), dim=0)
#     for j in range(num_iteration):
#         images.requires_grad = True
#         x_batch = torch.cat((img, img / 2., img / 4., img/ 8., img / 16.), dim=0)
#         outputs = model(input_diversity(x_batch))
#         if not isinstance(outputs, torch.Tensor):
#             outputs =outputs
#         loss = F.cross_entropy(outputs, y_batch)
#         grad_vanilla = torch.autograd.grad(loss, x_batch, grad_outputs=None, only_inputs=True)[0]
#         grad_batch_split = torch.split(grad_vanilla, split_size_or_sections=len(labels), dim=0)
#         grad_in_batch = torch.stack(grad_batch_split, dim=4)
#         new_grad = torch.sum(grad_in_batch * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
        
#         # current_grad = new_grad + variance
#         current_grad = new_grad
#         noise = F.conv2d(input=current_grad, weight=stack_kernel, stride=1, padding=3, groups=3)
#         noise = noise / torch.norm(noise, p=1)
#         adv = img + step_size  * noise.sign()
#         adv = torch.clamp(adv, 0.0, 1.0).detach()  
#         adv = torch.max(torch.min(adv, images+8/255), images-8/255).detach()
#         grads = noise
#         flag = (j+1) % check_point
#         if flag == 0:
#             point = j // check_point
#             suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(adv),dim=1) != labels).cpu().numpy()
#             suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(adv),dim=1) != labels).cpu().numpy()
#             suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(adv),dim=1) != labels).cpu().numpy()
#         if j == 49: 
#             save_images(adv.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)

# print('DI-TI-SI success rate:')
# print(suc/10000)

def dct(x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.fft.fft(v)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)
        return V


def idct(X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

        Our definition of idct is that idct(dct(x)) == x

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
        v = torch.fft.ifft(tmp)

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape).real
def dct_2d(x, norm=None):
        """
        2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last 2 dimensions
        """
        X1 = dct(x, norm=norm)
        X2 = dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
        """
        The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

        Our definition of idct is that idct_2d(dct_2d(x)) == x

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last 2 dimensions
        """
        x1 = idct(X, norm=norm)
        x2 = idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)
def clip_by_tensor(t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result


# ssa
save_dir = os.path.join('/data/crq/kmean','ssa',str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
suc = np.zeros((3,num_iteration // check_point))

for i, ((images, labels), path) in enumerate(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    x = images.clone()
    grad = 0
    rho = 0.5
    N = 20
    sigma = 4.0
    image_width=224
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    for i in range(num_iteration):
        noise = 0
        for n in range(N):
                gauss = torch.randn(x.size()[0], 3, image_width, image_width) * (sigma / 255)
                gauss = gauss.cuda()
                x_dct = dct_2d(x + gauss).cuda()
                mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
                x_idct = idct_2d(x_dct * mask)
                from torch.autograd import Variable as V
                x_idct = V(x_idct, requires_grad = True)
                output_v3 = model(x_idct)
                loss = F.cross_entropy(output_v3, labels)
                loss.backward()
                noise += x_idct.grad.data
        noise = noise / N
        images_min = clip_by_tensor(x - epsilon, 0.0, 1.0)
        images_max = clip_by_tensor(x + epsilon, 0.0, 1.0)
        x = x + step_size * torch.sign(noise)
        img = clip_by_tensor(x, images_min, images_max).detach()
        flag = (i+1) % check_point
        if flag == 0:
            point = i // check_point
            suc[0,point] = suc[0,point] + sum(torch.argmax(model_tar_1(img),dim=1) != labels).cpu().numpy()
            suc[1,point] = suc[1,point] + sum(torch.argmax(model_tar_2(img),dim=1) != labels).cpu().numpy()
            suc[2,point] = suc[2,point] + sum(torch.argmax(model_tar_3(img),dim=1) != labels).cpu().numpy()

        if i == 9: 
            save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)                       
print('ssa success rate:')
print(suc/1000)