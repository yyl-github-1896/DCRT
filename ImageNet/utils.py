
import numpy as np
import os
import pandas as pd
import random
import timm
import torch
import torch.nn as nn
import torchvision

from PIL import Image


# copy from advertorch
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
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

def jpeg2png(name):
    name_list = list(name)
    name_list[-4:-1] = 'png'
    name_list.pop(-1)
    return ''.join(name_list)


def save_images(adv, adv_path):
    '''
    save the adversarial images to the adv_path
    '''
    for idx, img in enumerate(adv):
        directory = os.path.dirname(adv_path[idx])
        if not os.path.exists(directory):
            os.makedirs(directory)
        torchvision.utils.save_image(img, adv_path[idx])
    return


def set_random_seed(seed=11037):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return


class MergedModel(nn.Module):
    def __init__(self, model_list=None, merge_mode=None):
        super(MergedModel, self).__init__()
        self.model_list = model_list
        self.num_models = len(model_list)   
        assert merge_mode in ['logits', 'softmax', 'sampling']
        self.merge_mode = merge_mode
        self.softmax = nn.Softmax(dim=1)
        print("self.merge_mode: {}".format(self.merge_mode))

    def forward(self, x):
        if self.merge_mode == 'softmax':
            out = self.softmax(self.model_list[0](x))
            for i in range(1, self.num_models):
                out = out + self.softmax(self.model_list[i](x))   # strange bug here
        elif self.merge_mode == 'logits':
            out = self.model_list[0](x)
            for i in range(1, self.num_models):
                out += self.model_list[i](x)
        elif self.merge_mode == 'sampling':
            import random
            seed = random.randint(0, self.num_models - 1)
            out = self.model_list[seed](x) 
        else:
            raise Exception('merge_mode {} not implemented!'.format(self.merge_mode))
        return out



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
import matplotlib.pyplot as plt
def avg(l):
    return sum(l) / len(l)

class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean).cuda()
        std = torch.as_tensor(std).cuda()
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

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
    
    mean_arr, stddev_arr = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    # denormalize
    for c in range(3):
        images[c, :, :] = images[c, :, :] * stddev_arr[c] + mean_arr[c]

    images = images.cpu().numpy()  # go from Tensor to numpy array
    # switch channel order back from
    # torch Tensor to PIL image: going from 3x32x128 - to 32x128x3
    images = np.transpose(images, (1, 2, 0))
    return images
from torchvision import transforms
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
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform, img_path, label_path):
        images = np.load(img_path)
        labels = np.load(label_path)
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3) or images.shape[1:] == (2, 32, 32, 3)
        if images.shape[1:] == (2, 32, 32, 3):
            clean_images = [Image.fromarray(x[0]) for x in images]
            adv_images = [Image.fromarray(x[1]) for x in images]
            self.images = [clean_images, adv_images]
        else:
            self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        if len(self.images) == 2:
            clean_image = self.images[0][index]
            clean_image = self.transform(clean_image)
            adv_image = self.images[1][index]
            adv_image = self.transform(adv_image)
            image = [clean_image, adv_image]
            label = self.labels[index]
        else:
            # print('index: {}, len(self.images): {}'.format(index, len(self.images)))    # test
            image, label = self.images[index], self.labels[index]
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)