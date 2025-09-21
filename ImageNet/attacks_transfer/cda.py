
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils2 import *
import random
seed=19
parser = argparse.ArgumentParser(description='Cross Data Transferability')
parser.add_argument('--train_dir', default='paintings', help='paintings, comics, imagenet')
parser.add_argument('--batch_size', type=int, default=30, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=4, help='Perturbation Budget')
parser.add_argument('--model_type', type=str, default='res50',
                    help='Model against GAN is trained: vgg16, vgg19, incv3, res152')
parser.add_argument('--attack_type', type=str, default='img', help='Training is either img/noise dependent')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
args = parser.parse_args()
print(args)

# Normalize (0-1)
eps = args.eps/255
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####################
# Model
#################### 
if args.model_type == 'vgg16':
    model = torchvision.models.vgg16(pretrained=True)
elif args.model_type == 'vgg19':
    model = torchvision.models.vgg19(pretrained=True)
elif args.model_type == 'incv3':
    model = torchvision.models.inception_v3(pretrained=True)
elif args.model_type == 'res152':
    model = torchvision.models.resnet152(pretrained=True)
elif args.model_type == 'res50':
    model = torchvision.models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

# Input dimensions
if args.model_type in ['vgg16', 'vgg19', 'res152','res50']:
    scale_size = 256
    img_size = 224
else:
    scale_size = 300
    img_size = 299


# Generator
if args.model_type == 'incv3':
    netG = GeneratorResnet(inception=True)
else:
     netG = GeneratorResnet()
netG.to(device)

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t

train_dir = '/data/crq/data/train'
import pandas as pd
from PIL import Image   
def jpeg2png(name):
    name_list = list(name)
    name_list[-4:-1] = 'png'
    name_list.pop(-1)
    return ''.join(name_list)
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir='/data/crq/data/val_rs.csv', transform=None, png=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.labels = pd.read_csv(self.label_dir).to_numpy()
        self.png = png

    def __getitem__(self, index):
        file_name, label = self.labels[index]
        label = torch.tensor(label) - 1
        file_dir = os.path.join(self.img_dir, file_name)
        if self.png:
            file_dir = jpeg2png(file_dir)
        img = Image.open(file_dir).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        output = img, label
        return output, file_name

    def __len__(self):
        return len(self.labels)
# img_dir = '/data/crq/data/val_clean'
# label_dir = '/data/crq/data/val_rs.csv'
img_dir='/data/crq/DRL/ImageNet/kmeans/img'
label_dir = '/data/crq/DRL/ImageNet/kmeans/km.csv'
val_set = MyDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        transform=data_transform,
        png=False,
    )
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False,
                                              num_workers=4)

train_set = datasets.ImageFolder(train_dir, data_transform)
# test_set = datasets.ImageFolder(test_dir, data_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=True, num_workers=4, pin_memory=True)
train_size = len(train_set)
test_size=len(val_set)
print('Training data size:', train_size)
print('Testing data size:', test_size)

# Loss
criterion = nn.CrossEntropyLoss()


####################
# Set-up noise if required
####################
if args.attack_type == 'noise':
    noise = np.random.uniform(0, 1, img_size * img_size * 3)
    # Save noise
    np.save('saved_models/noise_{}_{}_{}_rl'.format(args.target, args.model_type, args.train_dir), noise)
    im_noise = np.reshape(noise, (3, img_size, img_size))
    im_noise = im_noise[np.newaxis, :, :, :]
    im_noise_tr = np.tile(im_noise, (args.batch_size, 1, 1, 1))
    noise_tensor_tr = torch.from_numpy(im_noise_tr).type(torch.FloatTensor).to(device)


# Training
print('Label: {} \t Attack: {} dependent \t Model: {} \t Distribution: {} \t Saving instances: {}'.format(args.target, args.attack_type, args.model_type, args.train_dir, args.epochs))
for epoch in range(args.epochs):
    running_loss = 0
    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)

        # whatever the model think about the input
        label = model(normalize(img.clone().detach())).argmax(dim=-1).detach()
        
        if args.target == -1:
            targte_label = torch.LongTensor(img.size(0))
            targte_label.fill_(args.target)
            targte_label = targte_label.to(device)

        netG.train()
        optimG.zero_grad()

        if args.attack_type == 'noise':
            adv = netG(noise_tensor_tr)
        else:
            adv = netG(img)
        import torch.nn.functional as F
        adv = F.interpolate(adv, size=img.shape[2:], mode="bilinear", align_corners=False)
        # Projection
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        if args.target == -1:
            # Gradient accent (Untargetted Attack)
            adv_out = model(normalize(adv))
            img_out = model(normalize(img))

            loss = -criterion(adv_out-img_out, label)

        else:
            # Gradient decent (Targetted Attack)
            # loss = criterion(model(normalize(adv)), targte_label)
            loss = criterion(model(normalize(adv)), targte_label) + criterion(model(normalize(img)), label)
        loss.backward()
        optimG.step()

        if i % 1 == 0:
            print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 100))
            running_loss = 0
        running_loss += abs(loss.item())
        if i == 50:
            break
    torch.save(netG.state_dict(), '/data/crq/DRL/ImageNet/saved_models_CDA/netG_{}_{}_{}_{}_rl.pth'.format(args.target, args.attack_type, args.model_type, epoch))

    # Save noise
    if args.attack_type == 'noise':
        # Save transformed noise
        t_noise = netG(torch.from_numpy(im_noise).type(torch.FloatTensor).to(device))
        t_noise_np = np.transpose(t_noise[0].detach().cpu().numpy(), (1,2,0))
        f = plt.figure()
        plt.imshow(t_noise_np, interpolation='spline16')
        plt.xticks([])
        plt.yticks([])
        #plt.show()
        f.savefig('saved_models/noise_transformed_{}_{}_{}_{}_rl'.format(args.target, args.model_type, args.train_dir, epoch) + ".pdf", bbox_inches='tight')
        np.save('saved_models/noise_transformed_{}_{}_{}_{}_rl'.format(args.target, args.model_type, args.train_dir, epoch), t_noise_np)
def save_images(images, img_list, idx, output_dir):
    """Saves images to the output directory.
        Args:
          images: tensor with minibatch of images
          img_list: list of filenames 
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
          output_dir: directory where to save images
    """
    for i in range(idx):
        filename = os.path.basename(img_list[i])
        cur_images = (images[i, :, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)

        im = Image.fromarray(cur_images)
        im.save('{}.png'.format(os.path.join(output_dir, filename[:-5])))
if 1:
    netG.eval()
    correct_recon = 0
    correct_orig = 0
    fooled = 0
    total = 0
    label_count = {}

    for itr, ((images, class_label), path) in enumerate(val_loader):

        image = images.cuda()
        adv = netG(image).detach()
        adv = F.interpolate(adv, size=image.shape[2:], mode="bilinear", align_corners=False)
        adv = torch.min(torch.max(adv, image - eps), image + eps)
        adv = torch.clamp(adv, 0.0, 1.0)
        # if not os.path.exists(f"/data/crq/adv1000/CDA/res50"):
        #     os.mkdir(f"/data/crq/adv1000/CDA/res50")
        save_dir = os.path.join('/data/crq/kmean','cda',str(seed))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if  1:
            print('Images evaluated:', (itr*20))
            # undo normalize image color channels
            save_images(adv.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)
            # save_images(adv.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir='/data/crq/adv1000/CDA/res50')
            # for idx in range(adv.size(0)):
            #     # 提取样本的原始标签
            #     original_label = class_label[idx].item()  # 获取当前样本的原始标签
            #     # 保存对抗样本
            #     if original_label not in label_count:
            #         label_count[original_label] = 0
            #     # 获取当前标签下的样本编号
            #     current_idx = label_count[original_label]
            #     # 更新标签计数器
            #     label_count[original_label] += 1
            #     # print(label_count)
            #     torchvision.utils.save_image(adv[idx].detach(), 
            #                      f"/data/yyl/source/DRL_crq/ImageNet/out/CDA/res50eps8/{original_label}_{current_idx}.png")
            print('Saved images.')