from distutils.dep_util import newer_group
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# from utils2 import *
# from utils2 import cprint
from torch.autograd import Variable
from tqdm import tqdm
from argparse import Namespace
# import improved_diffusion
# # from improved_diffusion import dist_util
# from improved_diffusion.script_util import (
#                 NUM_CLASSES,
#                 model_and_diffusion_defaults,
#                 create_model_and_diffusion,
#                 add_dict_to_argparser,
#                 args_to_dict,
#                 add_dict_to_argparser,
#             )
def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

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


"""
adversarial attacks: 
Input: model, x, y, and other attack algorithm hyper-parameters 
Output: x, adv, is_adv
---------------------------------------------------------------
clean: No attack
white-box attack: fgsm, pgd, cw pgd. epsilon by default = 8/255
transfer attack: mi-fgsm, vmi-fgsm, vmi-ci,fgsm, fia. epsilon by default = 16/255
"""

def clean(model, x, y, epsilon=float(8/255), targeted=False):
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()
    # adv.requires_grad = True
    # outputs = model(adv)
    # loss = F.cross_entropy(outputs, y)
    # loss.backward()
    # adv = adv + epsilon * adv.grad.sign()
    # adv = torch.clamp(adv, 0.0, 1.0).detach()

    "validate in memory"
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    
    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def fgsm(model, x, y, epsilon=float(8/255), targeted=False):
    """
    reference: Goodfellow I J, Shlens J, Szegedy C. 
    Explaining and harnessing adversarial examples[J]. 
    arXiv preprint arXiv:1412.6572, 2014.
    """
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()
    adv.requires_grad = True
    outputs = model(adv)
    # print('outputs.shape: {}, y.shape: {}'.format(outputs.shape, y.shape))
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    if targeted:
        adv = adv - epsilon * adv.grad.sign()
    else:
        adv = adv + epsilon * adv.grad.sign()
    adv = torch.clamp(adv, 0.0, 1.0).detach()

    "validate in memory"
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    if y.dim() >= 2:
        '''switch one-hot label to class label'''
        target = y.topk(k=1, largest=True).indices
        is_adv = (pred_top1 == target) if targeted else (pred_top1 != target)
    else:
        is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def r_fgsm(model, x, y, epsilon=float(8/255), targeted=False):
    """
    reference: Tramèr F, Kurakin A, Papernot N, et al. Ensemble adversarial training: Attacks and defenses[C]. ICLR 2018.
    """
    x, y, model= x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()
    adv += (torch.rand_like(adv) * 2 - 1) * epsilon / 2
    adv = torch.clamp(adv, 0.0, 1.0).detach()

    min_x = x - epsilon
    max_x = x + epsilon

    adv.requires_grad = True
    outputs = model(adv)
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    if targeted:
        adv = adv - epsilon * adv.grad.sign()
    else:
        adv = adv + epsilon * adv.grad.sign()
        
    adv = torch.clamp(adv, 0.0, 1.0).detach()
    adv = torch.max(torch.min(adv, max_x), min_x).detach()

    "validate in memory"
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    if y.dim() >= 2:
        '''switch one-hot label to class label'''
        target = y.topk(k=1, largest=True).indices
        is_adv = (pred_top1 == target) if targeted else (pred_top1 != target)
    else:
        is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def pgd(model, x, y, epsilon=float(8/255), num_steps=20, step_size=float(2/255), targeted=False):
    """
    reference: Madry A, Makelov A, Schmidt L, et al. 
    Towards deep learning models resistant to adversarial attacks[J]. 
    arXiv preprint arXiv:1706.06083, 2017.
    """
    x, y, model= x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()
    adv += (torch.rand_like(adv) * 2 - 1) * epsilon
    adv = torch.clamp(adv, 0.0, 1.0).detach()

    min_x = x - epsilon
    max_x = x + epsilon

    for _ in range(num_steps):
        adv.requires_grad = True
        model.zero_grad()
        outputs = model(adv)
        loss = F.cross_entropy(outputs, y)
        loss.backward()
        if targeted:
            adv = adv - step_size * adv.grad.sign()
        else:
            adv = adv + step_size * adv.grad.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()

    '''validate in memory'''
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    if y.dim() >= 2:
        '''switch one-hot label to class label'''
        target = y.topk(k=1, largest=True).indices
        is_adv = (pred_top1 == target) if targeted else (pred_top1 != target)
    else:
        is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv

class Denoised_Classifier(torch.nn.Module):
    def __init__(self, diffusion, model, classifier, t):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.classifier = classifier
        self.t = t
    
    def sdedit(self, x, t, to_01=True):

        # assume the input is 0-1
        t_int = t
        
        x = x * 2 - 1
        
        t = torch.full((x.shape[0], ), t).long().to(x.device)
        # print(t)
        x_t = self.diffusion.q_sample(x, t) 
        
        sample = x_t
    
        # print(x_t.min(), x_t.max())
    
        # si(x_t, 'vis/noised_x.png', to_01=True)
        # print(range(t+1))
        indices = list(range(t_int+1))[::-1]

        
        # visualize 
        l_sample=[]
        l_predxstart=[]

        for i in indices:

            # out = self.diffusion.ddim_sample(self.model, sample, t)           
            out = self.diffusion.ddim_sample(self.model, sample, torch.full((x.shape[0], ), i).long().to(x.device))


            sample = out["sample"]


            l_sample.append(out['sample'])
            l_predxstart.append(out['pred_xstart'])
        
        
        # visualize
        si(torch.cat(l_sample), 'l_sample.png', to_01=1)
        si(torch.cat(l_predxstart), 'l_pxstart.png', to_01=1)

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2
        
        return sample
def get_imagenet_dm_conf(class_cond=False, respace="", device='cuda',
                         model_path='/data/yyl/source/DRL_crq/CIFAR-10/ckpt/dm/cifar10_uncond_50M_500K.pt'):
    # dist_util.setup_dist()
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )

    model_config = dict(
            use_fp16=False,
            diffusion_steps=4000,
            image_size=32,
            learn_sigma=True,
            num_res_blocks=3,
            dropout=0.3,
            lr=1e-4,
            noise_schedule='cosine',
            num_channels=128,
            batch_size=128,
        )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(model_config)
    args = Namespace(**defaults)
    

    model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    cprint('Create DM         ---------------', 'y')
    
    
    # load ckpt
    
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    
    cprint('Load DM Ckpt      ---------------', 'y')
    
    return model, diffusion
    
def diff_pgd(classifier, x, y,net, epsilon=float(8/255), num_steps=20, step_size=float(2/255), targeted=False):
    """
    reference: Madry A, Makelov A, Schmidt L, et al. 
    Towards deep learning models resistant to adversarial attacks[J]. 
    arXiv preprint arXiv:1706.06083, 2017.
    """
    y_pred=classifier(x).argmax(1)
    # model, diffusion = get_imagenet_dm_conf(respace='ddim50')
    # net = Denoised_Classifier(diffusion, model, classifier, t=3)

    
    delta = torch.zeros(x.shape).to(x.device)
    # delta.requires_grad_()

    loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")

    eps = float(8/255)
    alpha = float(2/255)
    iter = 10

    

    for pgd_iter_id in range(iter):
        
        # x_diff = net.sdedit(x+delta, t=3).detach()
        x_diff = net.sdedit(x+delta, t=3).detach()
        x_diff.requires_grad_()

        with torch.enable_grad():

            loss = loss_fn(classifier(x_diff), y_pred)

            loss.backward()

            grad_sign = x_diff.grad.data.sign()

        delta += grad_sign * alpha

        delta = torch.clamp(delta, -eps, eps)
    print("Done")

    x_adv = torch.clamp(x+delta, 0, 1)    
    outputs = classifier(x_adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    if y.dim() >= 2:
        '''switch one-hot label to class label'''
        target = y.topk(k=1, largest=True).indices
        is_adv = (pred_top1 == target) if targeted else (pred_top1 != target)
    else:
        is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)
    # 保存原始图像 x
    # save_tensor_as_image(x[0], '/data/yyl/source/DRL_crq/CIFAR-10/diff_pgd_result_example/original_image.jpg')

    # # 保存对抗性攻击后的图像 x_adv
    # save_tensor_as_image(x_adv[0], '/data/yyl/source/DRL_crq/CIFAR-10/diff_pgd_result_example/adversarial_image.jpg')
    return x,x_adv,is_adv
    # x, y, model= x.cuda(), y.cuda(), model.cuda().eval()

    # adv = x.clone()
    # adv += (torch.rand_like(adv) * 2 - 1) * epsilon
    # adv = torch.clamp(adv, 0.0, 1.0).detach()

    # min_x = x - epsilon
    # max_x = x + epsilon

    # for _ in range(num_steps):
    #     adv.requires_grad = True
    #     model.zero_grad()
    #     outputs = model(adv)
    #     loss = F.cross_entropy(outputs, y)
    #     loss.backward()
    #     if targeted:
    #         adv = adv - step_size * adv.grad.sign()
    #     else:
    #         adv = adv + step_size * adv.grad.sign()
    #     adv = torch.clamp(adv, 0.0, 1.0).detach()
    #     adv = torch.max(torch.min(adv, max_x), min_x).detach()

    # '''validate in memory'''
    # outputs = model(adv)
    # pred_top1 = outputs.topk(k=1, largest=True).indices
    # if pred_top1.dim() >= 2:
    #     pred_top1 = pred_top1.squeeze()

    # if y.dim() >= 2:
    #     '''switch one-hot label to class label'''
    #     target = y.topk(k=1, largest=True).indices
    #     is_adv = (pred_top1 == target) if targeted else (pred_top1 != target)
    # else:
    #     is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    # return x, adv, is_adv
import torchvision.transforms as transforms
from torchvision.utils import save_image

def save_tensor_as_image(tensor, filename):
    # 将张量转换为 PIL 图像
    tensor = tensor.clamp(0, 1)  # 将张量限制在 [0, 1] 范围内
    image = transforms.ToPILImage()(tensor.cpu())  # 转换为 PIL 图像
    image.save(filename)  # 保存为文件


def odi(model, X, y, epsilon=float(8/255), num_steps=20, step_size=float(2/255), ODI_num_steps=20, ODI_step_size=float(8/255), random=True, lossFunc='margin'):

    def margin_loss(logits,y):

        one_hot_y = torch.zeros(y.size(0), 10).cuda()
        one_hot_y[torch.arange(y.size(0)), y] = 1
        logit_org = logits.gather(1,y.view(-1,1))
        logit_target = logits.gather(1,(logits - one_hot_y * 9999).argmax(1, keepdim=True))
        loss = -logit_org + logit_target
        loss = torch.sum(loss)
        return loss

    X_pgd = Variable(X.data, requires_grad=True).cuda()

    randVector_ = torch.FloatTensor(*model(X_pgd).shape).uniform_(-1.,1.).cuda()
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for i in range(ODI_num_steps + num_steps):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            if y.dim() >= 2:
                # target = y.topk(k=1, largest=True).indices
                target = y
            else:
                target = y
            if i < ODI_num_steps:
                loss = (model(X_pgd) * randVector_).sum()
            elif lossFunc == 'xent':
                loss = nn.CrossEntropyLoss()(model(X_pgd), target)
            else:
                loss = margin_loss(model(X_pgd),target)
        loss.backward()
        if i < ODI_num_steps: 
            eta = ODI_step_size * X_pgd.grad.data.sign()
        else:
            eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    if y.dim() >= 2:
        '''switch one-hot label to class label'''
        target = y.data.topk(k=1, largest=True).indices
        is_adv = (model(X_pgd).data.max(1)[1] != target).detach().cpu().numpy() 
    else:
        is_adv = (model(X_pgd).data.max(1)[1] != y.data).detach().cpu().numpy() 
    return X, X_pgd.detach().clone(), is_adv


def cw_pgd(model, x, y, epsilon=float(8/255), num_steps=20, targeted=False):
    """
    pgd attack with cw loss, untargeted.
    reference: Carlini N, Wagner D. 
    Towards evaluating the robustness of neural networks[C]
    //2017 ieee symposium on security and privacy (sp). IEEE, 2017: 39-57.
    """
    x, y, model= x.cuda(), y.cuda(), model.cuda().eval()

    adv = x.clone()
    adv += (torch.rand_like(adv) * 2 - 1) * epsilon
    adv = torch.clamp(adv, 0.0, 1.0).detach()

    min_x = x - epsilon
    max_x = x + epsilon

    one_hot_y = torch.zeros(y.size(0), 10).cuda()
    one_hot_y[torch.arange(y.size(0)), y] = 1

    for _ in range(num_steps):
        adv.requires_grad = True
        outputs = model(adv)
        correct_logit = torch.sum(one_hot_y * outputs, dim=1)
        wrong_logit, _ = torch.max((1 - one_hot_y) * outputs - 1e4 * one_hot_y, dim=1)
        loss = -torch.sum(F.relu(correct_logit - wrong_logit + 50))
        loss.backward()
        if targeted:
            adv = adv - 0.00392 * adv.grad.sign()
        else:
            adv = adv + 0.00392 * adv.grad.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()

    '''validate in memory'''
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    if y.dim() >= 2:
        '''switch one-hot label to class label'''
        target = y.topk(k=1, largest=True).indices
        is_adv = (pred_top1 == target) if targeted else (pred_top1 != target)
    else:
        is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv



def mi_fgsm(model, x, y, epsilon=float(8/255), num_steps=10, targeted=False):
    """
    reference: Dong Y, Liao F, Pang T, et al. 
    Boosting adversarial attacks with momentum[C]//
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 9185-9193.
    """
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    alpha = epsilon / num_steps   # attack step size
    momentum = 1.0
    grads = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon

    adv = x.clone()

    for _ in range(num_steps):
        adv.requires_grad = True
        outputs = model(adv)
        loss = F.cross_entropy(outputs, y)
        loss.backward()
        new_grad = adv.grad
        noise = momentum * grads + (new_grad) / torch.norm(new_grad, dim=[1,2,3], p=1, keepdim=True)
        if targeted:
            adv = adv + alpha * noise.sign()
        else:
            adv = adv + alpha * noise.sign()

        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv


def vmi_fgsm(model, x, y, epsilon=float(16/255), targeted=False):
    """
    reference: Wang X, He K. 
    Enhancing the transferability of adversarial attacks through variance tuning[C]//
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 1924-1933.
    """
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    num_steps = 10
    alpha = epsilon / num_steps   # attack step size
    momentum = 1.0
    number = 20
    beta = 1.5
    grads = torch.zeros_like(x, requires_grad=False)
    variance = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon

    adv = x.clone()

    for _ in range(num_steps):
        adv.requires_grad = True
        outputs = model(adv)
        loss = F.cross_entropy(outputs, y)
        new_grad = torch.autograd.grad(loss, adv, grad_outputs=None, only_inputs=True)[0]
        noise = new_grad + variance
        noise = momentum * grads + noise / torch.norm(noise, p=1)

        # update variance
        sample = adv.clone().detach()
        global_grad = torch.zeros_like(x, requires_grad=False)
        for _ in range(number):
            sample = sample.detach()
            sample.requires_grad = True
            rd = (torch.rand_like(x) * 2 - 1) * beta * epsilon
            sample = sample + rd
            outputs_sample = model(sample)
            loss_sample = F.cross_entropy(outputs_sample, y)
            grad_vanilla_sample = torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]
            global_grad += grad_vanilla_sample
        variance = global_grad / (number * 1.0) - new_grad

        if targeted:
            adv = adv - alpha * noise.sign()
        else:
            adv = adv + alpha * noise.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()   # range [0, 1]
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()


    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv




def vmi_ci_fgsm(model, x, y, epsilon=float(16/255), dataset="ImageNet", targeted=False):
    """
    reference: Wang X, He K. 
    Enhancing the transferability of adversarial attacks through variance tuning[C]//
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 1924-1933.
    """
    def input_diversity(input_tensor):
        """apply input transformation to enhance transferability: padding and resizing (DIM)"""
        if dataset == "CIFAR":
            image_width = 32
            image_height = 32
            image_resize = 34
        
        elif dataset == "ImageNet":
            image_width = 299
            image_height = 299
            image_resize = 331
        prob = 0.5        # probability of using diverse inputs

        rnd = torch.randint(image_width, image_resize, ())   # uniform distribution
        rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='nearest')
        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = torch.randint(0, h_rem, ())
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem, ())
        pad_right = w_rem - pad_left
        # pad的参数顺序在pytorch里面是左右上下，在tensorflow里是上下左右，而且要注意pytorch的图像格式是BCHW, tensorflow是CHWB
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0))
        if torch.rand(1) < prob:
            ret = padded
        else:
            ret = input_tensor
        ret = F.interpolate(ret, [image_height, image_width], mode='nearest')
        return ret

    def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    kernel = gkern(7, 3).astype(np.float32)
    # 要注意Pytorch是BCHW, tensorflow是BHWC
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)  # batch, channel, height, width = 3, 1, 7, 7
    stack_kernel = torch.tensor(stack_kernel).cuda()

    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    num_steps = 10
    alpha = epsilon / num_steps   # attack step size
    momentum = 1.0
    number = 20
    beta = 1.5
    grads = torch.zeros_like(x, requires_grad=False)
    variance = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon

    adv = x.clone()
    y_batch = torch.cat((y, y, y, y, y), dim=0)

    for _ in range(num_steps):
        adv.requires_grad = True
        x_batch = torch.cat((adv, adv / 2., adv / 4., adv / 8., adv / 16.), dim=0)
        outputs = model(input_diversity(x_batch))
        loss = F.cross_entropy(outputs, y_batch)
        grad_vanilla = torch.autograd.grad(loss, x_batch, grad_outputs=None, only_inputs=True)[0]
        grad_batch_split = torch.split(grad_vanilla, split_size_or_sections=len(y), dim=0)
        grad_in_batch = torch.stack(grad_batch_split, dim=4)
        new_grad = torch.sum(grad_in_batch * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
        
        current_grad = new_grad + variance
        noise = F.conv2d(input=current_grad, weight=stack_kernel, stride=1, padding=3, groups=3)
        noise = momentum * grads + noise / torch.norm(noise, p=1)

        # update variance
        sample = x_batch.clone().detach()
        global_grad = torch.zeros_like(x, requires_grad=False)
        for _ in range(number):
            sample = sample.detach()
            sample.requires_grad = True
            rd = (torch.rand_like(x) * 2 - 1) * beta * epsilon
            rd_batch = torch.cat((rd, rd / 2., rd / 4., rd / 8., rd / 16.), dim=0)
            sample = sample + rd_batch
            outputs_sample = model(input_diversity(sample))
            loss_sample = F.cross_entropy(outputs_sample, y_batch)
            grad_vanilla_sample = torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]
            grad_batch_split_sample = torch.split(grad_vanilla_sample, split_size_or_sections=len(y),
                                                    dim=0)
            grad_in_batch_sample = torch.stack(grad_batch_split_sample, dim=4)
            global_grad += torch.sum(grad_in_batch_sample * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
        variance = global_grad / (number * 1.0) - new_grad

        if targeted:
            adv = adv - alpha * noise.sign()
        else:
            adv = adv + alpha * noise.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()  
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv
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
class Preprocessing_Layer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std

    def preprocess(self, img, mean, std):
        img = img.clone()
        #img /= 255.0

        img[:,0,:,:] = (img[:,0,:,:] - mean[0]) / std[0]
        img[:,1,:,:] = (img[:,1,:,:] - mean[1]) / std[1]
        img[:,2,:,:] = (img[:,2,:,:] - mean[2]) / std[2]

        #img = img.transpose(1, 3).transpose(2, 3)
        return(img)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res = self.preprocess(x, self.mean, self.std)
        return res

def sgm(
    model,
    images,
    labels,
    epsilon,
    step_size=8/2550,
    num_iteration=10
):
    from torchvision import models
    resnet = models.resnet50()
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    preprocess_layer = Preprocessing_Layer(mean,std)
    resnet = nn.Sequential(preprocess_layer, resnet)
    resnet.to(device)
    resnet.eval()
    gamma = 0.5
    register_hook_for_resnet(resnet, arch='resnet50', gamma=gamma)
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone().detach().requires_grad_(True)
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    for j in range(num_iteration):
        img_x = img.clone().detach().requires_grad_(True)
        att_out = resnet(img_x)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        resnet.zero_grad()
        loss.backward()
        # print(img_x.grad)
        input_grad = img_x.grad.data
        resnet.zero_grad()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)
    outputs = model(img)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 != labels)

    return images, img, is_adv

def GAP(
    model,
    netG,
    img,
    labels,
    epsilon,
    step_size=8/2550,
    num_iteration=10
):
    netG.eval()
    x = img.clone().detach().to(device)
    delta_im = netG(x)
    delta_im = delta_im + 1 # now 0..2
    delta_im = delta_im * 0.5
    delta_im = delta_im[:, :, :x.size(2), :x.size(3)]
    mean_arr = [0.4914, 0.4822, 0.4465]
    stddev_arr = [0.2023, 0.1994, 0.2010]
    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    bs = delta_im.size(0)
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(3):
            # print(delta_im.size())
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = 4/(255.0*stddev_arr[ci])
            gpu_id = device.index
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())
    recons = torch.add(x.to(device), delta_im[0:x.size(0)].to(device)).cuda()
    recons = torch.clamp(recons, min=0, max=1)
    outputs = model(recons)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv =  (pred_top1 != labels)

    return img, recons, is_adv

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean).cuda()
        std = torch.as_tensor(std).cuda()
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
invNormalize = NormalizeInverse([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
Normlize_Trans = transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
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
def NAA_attack(model,
    X,    
    y,    
    feature_layer,
    N=30,    
    niters=10,    
    epsilon=0.01,
    learning_rate=1,
    decay=1,        
    dataset="cifar10",
    use_Inc_model = False,):
    """CIFAR-10版多样化输入"""
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
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)

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
def NAA(
    model,
    images,
    labels,
    epsilon,
    step_size=8/2550,
    num_iteration=10
):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone().detach().requires_grad_(True)
    resnet50_model = model._modules.get('1')
    source_layers = get_source_layers(model_name='ResNet50', model=resnet50_model)
    adversaries = NAA_attack(model, img, labels, source_layers[2][1][1], learning_rate=step_size, epsilon=epsilon, niters=num_iteration, dataset='cifar10')
    adversaries = torch.clamp(adversaries, min=0, max=1)
    outputs = model(adversaries)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 != labels)

    return images, adversaries, is_adv

def DI(input_tensor):
    """CIFAR-10版多样化输入"""
    prob = 0.5        # probability of using diverse inputs
    image_width = 32
    image_height = 32
    image_resize = 34
    rnd = torch.randint(image_width, image_resize, ())   # uniform distribution
    rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='nearest')
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    # pad的参数顺序在pytorch里面是左右上下，在tensorflow里是上下左右，而且要注意pytorch的图像格式是BCHW, tensorflow是CHWB
    padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0))
    if torch.rand(1) < prob:
            ret = padded
    else:
            ret = input_tensor
    ret = F.interpolate(ret, [image_height, image_width], mode='nearest')
    return ret
def di_fgsm(
    model,
    images,
    labels,
    epsilon,
    step_size=8/2550,
    num_iteration=10
):
    images = images.to(device)
    labels = labels.to(device)
    img = images.clone().detach().requires_grad_(True)
    model.to(device)
    multi_copies=5
    for j in range(num_iteration):
        img_x = img.clone().detach().requires_grad_(True)
        if not multi_copies:
            logits = model(DI(img_x))
            loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
            loss.backward()
            input_grad = img_x.grad.clone()
        else:               
            input_grad = 0
            for c in range(multi_copies):
                logits = model(DI(img_x))
                loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
                loss.backward()
                input_grad = input_grad + img_x.grad.clone()     
                img_x.grad.zero_()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)
    outputs = model(img)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 != labels)

    return images, img, is_adv


def ci_fgsm(model, x, y, epsilon=float(16/255), dataset="cifar10", targeted=False):
    """
    reference: Wang X, He K. 
    Enhancing the transferability of adversarial attacks through variance tuning[C]//
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 1924-1933.
    """
    def input_diversity(input_tensor):
        """apply input transformation to enhance transferability: padding and resizing (DIM)"""
        if dataset == "cifar10":
            image_width = 32
            image_height = 32
            image_resize = 34
        
        elif dataset == "imagenet":
            image_width = 299
            image_height = 299
            image_resize = 331
        prob = 0.5        # probability of using diverse inputs

        rnd = torch.randint(image_width, image_resize, ())   # uniform distribution
        rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='nearest')
        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = torch.randint(0, h_rem, ())
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem, ())
        pad_right = w_rem - pad_left
        # pad的参数顺序在pytorch里面是左右上下，在tensorflow里是上下左右，而且要注意pytorch的图像格式是BCHW, tensorflow是CHWB
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0))
        if torch.rand(1) < prob:
            ret = padded
        else:
            ret = input_tensor
        ret = F.interpolate(ret, [image_height, image_width], mode='nearest')
        return ret

    def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    kernel = gkern(7, 3).astype(np.float32)
    # 要注意Pytorch是BCHW, tensorflow是BHWC
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)  # batch, channel, height, width = 3, 1, 7, 7
    stack_kernel = torch.tensor(stack_kernel).cuda()

    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    num_steps = 10
    alpha = epsilon / num_steps   # attack step size
    momentum = 1.0
    number = 20
    beta = 15
    grads = torch.zeros_like(x, requires_grad=False)
    # variance = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon

    adv = x.clone()
    y_batch = torch.cat((y, y, y, y, y), dim=0)


    for _ in range(num_steps):
        adv.requires_grad = True
        x_batch = torch.cat((adv, adv / 2., adv / 4., adv / 8., adv / 16.), dim=0)
        outputs = model(input_diversity(x_batch))
        loss = F.cross_entropy(outputs, y_batch)
        grad_vanilla = torch.autograd.grad(loss, x_batch, grad_outputs=None, only_inputs=True)[0]
        grad_batch_split = torch.split(grad_vanilla, split_size_or_sections=len(y), dim=0)
        grad_in_batch = torch.stack(grad_batch_split, dim=4)
        new_grad = torch.sum(grad_in_batch * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
        
        # current_grad = new_grad + variance
        current_grad = new_grad
        noise = F.conv2d(input=current_grad, weight=stack_kernel, stride=1, padding=3, groups=3)
        noise = momentum * grads + noise / torch.norm(noise, p=1)
        
        if targeted:
            adv = adv - alpha * noise.sign()
        else:
            adv = adv + alpha * noise.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()  
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        grads = noise

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv
def fia(
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
    X, y, model = X.cuda(), y.cuda(), model.cuda().eval()
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
    X_pert= torch.clamp(X_pert, min=0, max=1)   
    # return adversaries 
    output = model(X_pert)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    return X, X_pert.clone().detach(), (pred_top1 != y)   

class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean).cuda()
        std = torch.as_tensor(std).cuda()
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
    
#     return x, adv, (pred_top1 != y)
invNormalize = NormalizeInverse([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
Normlize_Trans = transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])

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
mid_output = None

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
    use_Inc_model = True,
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
    output = model(X_pert)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    return X,X_pert.clone().detach(),(pred_top1 != y)
    # return adversaries

# square sum of dot product
class Proj_Loss(torch.nn.Module):
    def __init__(self):
        super(Proj_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)
        x_norm = x / x.norm()

        proj_loss = torch.mm(y, x_norm.transpose(0, 1)) / x.norm()
        return proj_loss


# square sum of dot product
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

# TAP (transferable adversairal perturbation ECCV 2018)
class Transferable_Adversarial_Perturbations_Loss(torch.nn.Module):
    def __init__(self):
        super(Transferable_Adversarial_Perturbations_Loss, self).__init__()

    def forward(
        self,
        X,
        X_pert,
        original_mids,
        new_mids,
        y,
        output_perturbed,
        lam,
        alpha,
        s,
        yita,
    ):

        l1 = nn.CrossEntropyLoss()(output_perturbed, y)

        l2 = 0
        for i, new_mid in enumerate(new_mids):
            a = torch.sign(original_mids[i]) * torch.pow(
                torch.abs(original_mids[i]), alpha
            )
            b = torch.sign(new_mid) * torch.pow(torch.abs(new_mid), alpha)
            l2 += lam * (a - b).norm() ** 2

        l3 = yita * torch.abs(nn.AvgPool2d(s)(X - X_pert)).sum()

        return l1 + l2 + l3


mid_outputs = []
mid_grads = []

def Transferable_Adversarial_Perturbations(
    model,
    X,
    y,
    niters=10,
    epsilon=0.03,
    lam=0.005,
    alpha=0.5,
    s=3,
    yita=0.01,
    learning_rate=0.006,
    dataset="cifar10",
    use_Inc_model = True,
):
    """Perform cifar10 TAP attack using model on images X with labels y

    Args:
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        y: labels corresponding to the batch of images
        niters: number of iterations of TAP to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        lam: lambda parameter of TAP
        alpha: alpha parameter of TAP
        s: s parameter of TAP
        yita: yita parameter of TAP
        learning_rate: learning rate of TAP attack

    Returns:
        The batch of adversarial examples corresponding to the original images
    """
    feature_layers = list(model._modules.keys())
    global mid_outputs
    y=y.detach().cuda()
    X = X.detach().cuda()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach().cuda()
    X_pert.requires_grad = True
    model.cuda()
    def get_mid_output(m, i, o):
        global mid_outputs
        mid_outputs.append(o)

    hs = []
    for layer_name in feature_layers:
        hs.append(model._modules.get(layer_name).register_forward_hook(get_mid_output))

    out = model(X)
        
    mid_originals = []
    for mid_output in mid_outputs:
        mid_original = torch.zeros(mid_output.size()).cuda()
        mid_originals.append(mid_original.copy_(mid_output))

    mid_outputs = []
    adversaries = []

    for iter_n in range(niters):
        output_perturbed = model(X_pert)
        # generate adversarial example by max middle
        # layer pertubation in the direction of increasing loss
        mid_originals_ = []
        for mid_original in mid_originals:
            mid_originals_.append(mid_original.detach())

        loss = Transferable_Adversarial_Perturbations_Loss()(
            X,
            X_pert,
            mid_originals_,
            mid_outputs,
            y,
            output_perturbed,
            lam,
            alpha,
            s,
            yita,
        )
        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

        mid_outputs = []

        if (iter_n+1) % 10 == 0:
            adversaries.append(X_pert.clone().detach())

    for h in hs:
        h.remove()
    output = model(X_pert)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    return X,X_pert.clone().detach(),(pred_top1 != y)

def AA(
    model,
    X,    
    y,  
    X_target,  
    feature_layer,        
    niters=10,    
    epsilon=0.01,
    learning_rate=1,
    decay=1,
    dataset="cifar10",
    use_Inc_model = True,    
):
    """
        Activation Attack
    """
    batch_size = X.shape[0]
    X = X.detach()
    X_target = X_target.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o            

    h = feature_layer.register_forward_hook(get_mid_output)    

    '''
        Set Seeds
    '''
    torch.manual_seed(0)
    np.random.seed(0)

    '''
     Select target for each input
    '''
    with torch.no_grad():
        model(X)
        X_mid_output = mid_output.clone().detach()
        model(X_target)
        X_target_mid_output = mid_output.clone().detach()
    target_slice = []
    for X_i in X_mid_output:
        max_JAA = 0
        max_slice = 0
        for slice_i, X_target_i in enumerate(X_target_mid_output):
            JAA = (X_i-X_target_i).norm(2).item()
            if JAA>max_JAA:
                max_JAA = JAA
                max_slice = slice_i
        target_slice.append(X_target_mid_output[max_slice].clone().detach())
    targets = torch.stack(target_slice)


    adversaries = []

    momentum = 0
    for iter_n in range(niters):
        output_perturbed = model(X_pert)
        loss = (mid_output - targets).reshape(batch_size, -1).norm(2, dim=1).sum()
        model.zero_grad()
        loss.backward()

        # momentum = decay * momentum + X_pert.grad / torch.sum(torch.abs(X_pert.grad))
        # pert = learning_rate * momentum.sign()

        # No Momentum
        pert = -learning_rate * X_pert.grad.detach().sign()

        X_pert = update_adv(X, X_pert, pert, epsilon)
        X_pert.requires_grad = True        

        if (iter_n+1) % 10 == 0:
            adversaries.append(X_pert.clone().detach())

    h.remove()        
    # return adversaries    
    output = model(X_pert)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    return X,X_pert.clone().detach(),(pred_top1 != y)

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
    use_Inc_model = True,    
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
    # print(mid_grad)
    '''
        Set Seeds
    '''
    torch.manual_seed(0)
    np.random.seed(0)

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
    output = model(X_pert)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    return X,X_pert.clone().detach(),(pred_top1 != y)

class Preprocessing_Layer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std

    def preprocess(self, img, mean, std):
        img = img.clone()
        #img /= 255.0

        img[:,0,:,:] = (img[:,0,:,:] - mean[0]) / std[0]
        img[:,1,:,:] = (img[:,1,:,:] - mean[1]) / std[1]
        img[:,2,:,:] = (img[:,2,:,:] - mean[2]) / std[2]

        #img = img.transpose(1, 3).transpose(2, 3)
        return(img)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res = self.preprocess(x, self.mean, self.std)
        return res
    
def SGM(model, x, y, epsilon=float(8/255), mi=True,dataset="cifar10", device=None):

    epsilon = 8/255
    # step_size = 2.0 / 255.0
    num_iteration = 10
    step_size = epsilon/num_iteration
    check_point = 5
    multi_copies = 5
    momentum = 1.0
    grads = 0
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()
    images = x
    labels = y
    from torchvision import models
    resnet = models.resnet50()
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
    preprocess_layer = Preprocessing_Layer(mean,std)
    resnet = nn.Sequential(preprocess_layer, resnet)
    resnet.cuda()
    suc = np.zeros((3, num_iteration // check_point))
    # model=model.cuda()
    # images = images.cuda()
    # labels = labels.cuda()
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out = resnet(img_x)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        resnet.zero_grad()
        loss.backward()
        input_grad = img_x.grad.data
        resnet.zero_grad()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)
    adv = img

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)

def linbp_forw_resnet50(model, x, do_linbp, linbp_layer):
    jj = int(linbp_layer.split('_')[0])
    kk = int(linbp_layer.split('_')[1])
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    x = model[1].maxpool(x)
    ori_mask_ls = []
    conv_out_ls = []
    relu_out_ls = []
    conv_input_ls = []
    def layer_forw(jj, kk, jj_now, kk_now, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp):
        if jj < jj_now:
            x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
            ori_mask_ls.append(ori_mask)
            conv_out_ls.append(conv_out)
            relu_out_ls.append(relu_out)
            conv_input_ls.append(conv_in)
        elif jj == jj_now:
            if kk_now >= kk:
                x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
                ori_mask_ls.append(ori_mask)
                conv_out_ls.append(conv_out)
                relu_out_ls.append(relu_out)
                conv_input_ls.append(conv_in)
            else:
                x, _, _, _, _ = block_func(mm, x, linbp=False)
        else:
            x, _, _, _, _ = block_func(mm, x, linbp=False)
        return x, ori_mask_ls
    for ind, mm in enumerate(model[1].layer1):
        x, ori_mask_ls = layer_forw(jj, kk, 1, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer2):
        x, ori_mask_ls = layer_forw(jj, kk, 2, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer3):
        x, ori_mask_ls = layer_forw(jj, kk, 3, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer4):
        x, ori_mask_ls = layer_forw(jj, kk, 4, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    x = model[1].avgpool(x)
    x = torch.flatten(x, 1)
    x = model[1].fc(x)
    return x, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls

def block_func(block, x, linbp):
    identity = x
    conv_in = x+0
    out = block.conv1(conv_in)
    out = block.bn1(out)
    out_0 = out + 0
    if linbp:
        out = linbp_relu(out_0)
    else:
        out = block.relu(out_0)
    ori_mask_0 = out.data.bool().int()

    out = block.conv2(out)
    out = block.bn2(out)
    out_1 = out + 0
    if linbp:
        out = linbp_relu(out_1)
    else:
        out = block.relu(out_1)
    ori_mask_1 = out.data.bool().int()

    out = block.conv3(out)
    out = block.bn3(out)

    if block.downsample is not None:
        identity = block.downsample(identity)
    identity_out = identity + 0
    x_out = out + 0


    out = identity_out + x_out
    out = block.relu(out)
    ori_mask_2 = out.data.bool().int()
    return out, (ori_mask_0, ori_mask_1, ori_mask_2), (identity_out, x_out), (out_0, out_1), (0, conv_in)

def linbp_relu(x):
    x_p = F.relu(-x)
    x = x + x_p.data
    return x

def linbp_backw_resnet50(img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp):
    for i in range(-1, -len(conv_out_ls)-1, -1):
        if i == -1:
            grads = torch.autograd.grad(loss, conv_out_ls[i])
        else:
            grads = torch.autograd.grad((conv_out_ls[i+1][0], conv_input_ls[i+1][1]), conv_out_ls[i], grad_outputs=(grads[0], main_grad_norm))
        normal_grad_2 = torch.autograd.grad(conv_out_ls[i][1], relu_out_ls[i][1], grads[1]*ori_mask_ls[i][2],retain_graph=True)[0]
        normal_grad_1 = torch.autograd.grad(relu_out_ls[i][1], relu_out_ls[i][0], normal_grad_2 * ori_mask_ls[i][1], retain_graph=True)[0]
        normal_grad_0 = torch.autograd.grad(relu_out_ls[i][0], conv_input_ls[i][1], normal_grad_1 * ori_mask_ls[i][0], retain_graph=True)[0]
        del normal_grad_2, normal_grad_1
        main_grad = torch.autograd.grad(conv_out_ls[i][1], conv_input_ls[i][1], grads[1])[0]
        alpha = normal_grad_0.norm(p=2, dim = (1,2,3), keepdim = True) / main_grad.norm(p=2,dim = (1,2,3), keepdim=True)
        main_grad_norm = xp * alpha * main_grad
    input_grad = torch.autograd.grad((conv_out_ls[0][0], conv_input_ls[0][1]), img, grad_outputs=(grads[0], main_grad_norm))
    return input_grad[0].data

def LinBP(model, x, y, epsilon=float(8/255), mi=True,dataset="cifar10", device=None):

    epsilon = 8/255
    # step_size = 2.0 / 255.0
    num_iteration = 10
    step_size = epsilon/num_iteration
    check_point = 5
    multi_copies = 5
    momentum = 1.0
    grads = 0
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()
    images = x
    labels = y
    from torchvision import models
    resnet = models.resnet50().cuda()
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
    preprocess_layer = Preprocessing_Layer(mean,std)
    resnet = nn.Sequential(preprocess_layer, resnet)
    suc = np.zeros((3, num_iteration // check_point))
    linbp_layer = '3_1'
    sgm_lambda = 1.0
    # model=model.cuda()
    # images = images.cuda()
    # labels = labels.cuda()
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(resnet, img_x, True, linbp_layer)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        resnet.zero_grad()
        input_grad = linbp_backw_resnet50(img_x, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=sgm_lambda)
        resnet.zero_grad()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)
    adv = img

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)

def RFA(model, x, y, epsilon=float(8/255), mi=True,dataset="cifar10", device=None):

    epsilon = 8/255
    # step_size = 2.0 / 255.0
    num_iteration = 10
    step_size = epsilon/num_iteration
    check_point = 5
    multi_copies = 5
    momentum = 1.0
    grads = 0
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()
    images = x
    labels = y
    from torchvision import models
    resnet = models.resnet50()
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
    preprocess_layer = Preprocessing_Layer(mean,std)
    resnet = nn.Sequential(preprocess_layer, resnet)
    resnet.to(device)
    resnet.eval()
    suc = np.zeros((3, num_iteration // check_point))
    model=model.cuda()
    images = images.cuda()
    labels = labels.cuda()
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out = resnet(img_x)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        resnet.zero_grad()
        loss.backward()
        input_grad = img_x.grad.data
        resnet.zero_grad()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)
    adv = img

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)

def IAA(model,resnet, x, y, epsilon=float(8/255), mi=True,dataset="cifar10", device=None):

    epsilon = 8/255
    # step_size = 2.0 / 255.0
    num_iteration = 10
    step_size = epsilon/num_iteration
    check_point = 5
    multi_copies = 5
    momentum = 1.0
    grads = 0
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()
    images = x
    labels = y
    # from ops_attack.utils_iaa import resnet50
    # resnet = resnet50()
    # model_path = '/data/crq/DRL/ImageNet/models/resnet50-0676ba61.pth'
    # pre_dict = torch.load(model_path)
    # resnet_dict = resnet.state_dict()
    # state_dict = {k:v for k,v in pre_dict.items() if k in resnet_dict.keys()}
    # print("loaded pretrained weight. Len:",len(pre_dict.keys()),len(state_dict.keys()))
    # resnet_dict.update(state_dict)
    # model_dict = resnet.load_state_dict(resnet_dict)
    # mean=[0.4914, 0.4822, 0.4465]
    # std=[0.2023, 0.1994, 0.2010]
    # preprocess_layer = Preprocessing_Layer(mean,std)
    # resnet = nn.Sequential(preprocess_layer, resnet)
    # resnet.to(device)
    # resnet.eval()
    # model=model.cuda()
    # images = images.cuda()
    # labels = labels.cuda()
    img = images.clone()
    for j in range(num_iteration):
        img_x = img
        img_x.requires_grad_(True)
        att_out = resnet(img_x)
        pred = torch.argmax(att_out, dim=1).view(-1)
        loss = nn.CrossEntropyLoss()(att_out, labels)
        resnet.zero_grad()
        loss.backward()
        input_grad = img_x.grad.data
        resnet.zero_grad()
        img = img.data + step_size * torch.sign(input_grad)
        img = torch.where(img > images + epsilon, images + epsilon, img)
        img = torch.where(img < images - epsilon, images - epsilon, img)
        img = torch.clamp(img, min=0, max=1)
    adv = img

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)

# import torch
# import numpy as np
# from attacks_transfer.material.models.generators import ResnetGenerator

def BIA(model, x, y,netG,epsilon=float(8/255), arch="vgg16", dataset='cifar10', device=None, sub_batch_size=20):
    """
    Generate adversarial examples in smaller batches to prevent memory overflow.
    """
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    # # Initialize the generator
    # netG = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=[0,1])
    # checkpoint='/data/yyl/source/DRL_crq/ImageNet/netG_model_epoch_10_'
    # # print("=> loading checkpoint '{}'".format(checkpoint))
    # # state_dict = torch.load(checkpoint)["state_dict"]
    # # new_state_dict = {}
    # # for k, v in state_dict.items():
    # #     # 去除module.前缀
    # #     if k=='normalize.mean' or k=='normalize.std' or k=='module.0.mean' or k=='module.0.std':
    # #         1
    # #     else:  
    # #         name = k.replace("module.1._wrapped_model.", "module.")
    # #         # name = k.replace("model.", "module.")
    # #         new_state_dict[name] = v
    # # model.load_state_dict(new_state_dict)
    # netG.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
    # print("=> loaded checkpoint '{}'".format(checkpoint))
    # netG.eval()

    # Split the input `x` into smaller batches
    # image = x.cuda()
    adv = netG(x).detach()
    import torch.nn.functional as F
    adv = F.interpolate(adv, size=x.shape[2:], mode="bilinear", align_corners=False)
    adv = torch.min(torch.max(adv, x - epsilon), x + epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)      
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices

    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)

def GAP2(model, x, y,netG,epsilon=float(8/255), arch="vgg16", dataset='cifar10', device=None, sub_batch_size=10):
    """
    Generate adversarial examples in smaller batches to prevent memory overflow.
    """
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    # # Initialize the generator
    # netG = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=[0,1])
    # checkpoint='/data/yyl/source/DRL_crq/ImageNet/netG_model_epoch_10_'
    # # print("=> loading checkpoint '{}'".format(checkpoint))
    # # state_dict = torch.load(checkpoint)["state_dict"]
    # # new_state_dict = {}
    # # for k, v in state_dict.items():
    # #     # 去除module.前缀
    # #     if k=='normalize.mean' or k=='normalize.std' or k=='module.0.mean' or k=='module.0.std':
    # #         1
    # #     else:  
    # #         name = k.replace("module.1._wrapped_model.", "module.")
    # #         # name = k.replace("model.", "module.")
    # #         new_state_dict[name] = v
    # # model.load_state_dict(new_state_dict)
    # netG.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
    # print("=> loaded checkpoint '{}'".format(checkpoint))
    # netG.eval()

    # Split the input `x` into smaller batches
    batch_size = x.size(0)
    recons_list = []

    for i in range(0, batch_size, sub_batch_size):
        x_sub = x[i:i + sub_batch_size].to(device)
        y_sub = y[i:i + sub_batch_size].to(device)

        # Generate adversarial perturbations for the sub-batch
        with torch.no_grad():
            delta_im = netG(x_sub)
            delta_im = (delta_im + 1) * 0.5  # Normalize to [0, 1]
            delta_im = delta_im[:, :, :x_sub.size(2), :x_sub.size(3)]

            bs = delta_im.size(0)
            mean_arr = [0.4914, 0.4822, 0.4465]
            stddev_arr = [0.2023, 0.1994, 0.2010]
            for i in range(bs):
                # do per channel l_inf normalization
                for ci in range(3):
                    # print(delta_im.size())
                    l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
                    mag_in_scaled_c = 4/(255.0*stddev_arr[ci])
                    gpu_id = device.index
                    delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())
            recons = torch.add(x_sub.to(device), delta_im[0:x_sub.size(0)].to(device)).cuda()
            for cii in range(3):
                recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(x_sub[:,cii,:,:].min(), x_sub[:,cii,:,:].max())
            recons_list.append(recons)
    recons = torch.cat(recons_list, dim=0)        
    output = model(recons)
    pred_top1 = output.topk(k=1, largest=True).indices

    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, recons, (pred_top1 != y)

def GAP(model,netG, x, y, epsilon=float(8/255), arch="vgg16", dataset='cifar10', device=None):
    # from attacks_transfer.material.models.generators import ResnetGenerator, weights_init
    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()
    # gpu_id=[device.index]
    # netG = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=[0])
    # checkpoint='/data/crq/tempname/netG_model_epoch_10_'
    # print("=> loading checkpoint '{}'".format(checkpoint))
    # state_dict = torch.load(checkpoint)["state_dict"]
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     # 去除module.前缀
    #     if k=='normalize.mean' or k=='normalize.std' or k=='module.0.mean' or k=='module.0.std':
    #         1
    #     else:  
    #         name = k.replace("module.1._wrapped_model.", "module.")
    #         # name = k.replace("model.", "module.")
    #         new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    # netG.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
    # print("=> loaded checkpoint '{}'".format(checkpoint))
    netG.eval()
    delta_im = netG(x)
    delta_im = delta_im + 1 # now 0..2
    delta_im = delta_im * 0.5
    delta_im = delta_im[:, :, :x.size(2), :x.size(3)]
    mean_arr = [0.4914, 0.4822, 0.4465]
    stddev_arr = [0.2023, 0.1994, 0.2010]
    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    bs = delta_im.size(0)
    mean_arr = [0.4914, 0.4822, 0.4465]
    stddev_arr = [0.2023, 0.1994, 0.2010]
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(3):
            # print(delta_im.size())
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = 4/(255.0*stddev_arr[ci])
            gpu_id = device.index
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())
    recons = torch.add(x.to(device), delta_im[0:x.size(0)].to(device)).cuda()
    for cii in range(3):
        recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(x[:,cii,:,:].min(), x[:,cii,:,:].max())
    output = model(recons)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, recons, (pred_top1 != y)

class GeneratorResnet(nn.Module):
    def __init__(self, inception=False, data_dim='high'):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        self.data_dim = data_dim
        ngf = 64
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
            # self.resblock7 = ResidualBlock(ngf*4)
            # self.resblock8 = ResidualBlock(ngf*4)
            # self.resblock9 = ResidualBlock(ngf*4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.data_dim == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
            # x = self.resblock7(x)
            # x = self.resblock8(x)
            # x = self.resblock9(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        return (torch.tanh(x) + 1) / 2 # Output range [0 1]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual

def GAPF(model, x, y, epsilon=float(8/255), arch="vgg16", dataset='cifar10', device=None):    

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    netG = GeneratorResnet()
    netG.load_state_dict(torch.load('saved_models/netG_{}_{}_{}_{}_{}.pth'))
    # Load Generator
    netG.to(device)
    netG.eval()
    eps = 8/255
    x = x.to(device)
    adv = netG(x).detach()
    adv = torch.min(torch.max(adv, x - eps), x + eps)
    adv = torch.clamp(adv, 0.0, 1.0)
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)

def rpa(model, x, y, epsilon=float(16/255), arch="vgg16", dataset='cifar10'):
    '''
    Reference: Zhang Y, Tan Y, Chen T, et al. Enhancing the Transferability of Adversarial Examples with Random Patch[C] IJCAI'21.
    '''
    num_iter = 10
    alpha = epsilon / num_iter
    momentum = 1.0

    if dataset == 'cifar10':
        num_classes = 10
        image_size = 32
    elif dataset == 'imagenet':
        num_classes = 1000
        image_size = 299
    ens = 60
    probb = 0.7

    gradients = torch.zeros_like(x, requires_grad=False)
    min_x = x - epsilon
    max_x = x + epsilon

    batch_shape = [len(y), 3, image_size, image_size]

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    if arch == "vgg16_fp":
        model[1].features[15].register_forward_hook(get_activation("features"))
    elif arch == "vgg16_bn_fp":
        model[1].features[22].register_forward_hook(get_activation("features"))
    elif arch == "vgg16_qdrop":
        model[1].model.features[15].register_forward_hook(get_activation("features"))
    elif arch == 'resnet20_fp' or arch == 'resnet56_fp' or arch == 'resnet50_fp':
        model[1].module.layer2.register_forward_hook(get_activation("features"))
    elif arch == 'resnet18_qdrop' or arch == 'resnet50_qdrop':
        model[1].model.layer2.register_forward_hook(get_activation("features"))
    elif arch == 'resnet20_apot' or arch == 'resnet56_apot':
        model[1].module.layer2.register_forward_hook(get_activation("features"))
    else:
        raise Exception("arch {} not implemented!".format(arch))


    def patch_by_strides(img_shape, patch_size, prob):
        img_shape = (img_shape[0], img_shape[2], img_shape[3], img_shape[1])  # from pytorch (BCHW) to tf (BHWC)
        X_mask = np.ones(img_shape)
        N0, H0, W0, C0 = X_mask.shape
        ph = H0 // patch_size[0]
        pw = W0 // patch_size[1]
        X = X_mask[:, :ph * patch_size[0], :pw * patch_size[1]]
        N, H, W, C = X.shape
        shape = (N, ph, pw, patch_size[0], patch_size[1], C)
        strides = (X.strides[0], X.strides[1] * patch_size[0], X.strides[2] * patch_size[0], *X.strides[1:])
        mask_patchs = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        mask_len = mask_patchs.shape[1] * mask_patchs.shape[2] * mask_patchs.shape[-1]
        ran_num = int(mask_len * (1 - prob))
        rand_list = np.random.choice(mask_len, ran_num, replace=False)
        for i in range(mask_patchs.shape[1]):
            for j in range(mask_patchs.shape[2]):
                for k in range(mask_patchs.shape[-1]):
                    if i * mask_patchs.shape[2] * mask_patchs.shape[-1] + j * mask_patchs.shape[-1] + k in rand_list:
                        mask_patchs[:, i, j, :, :, k] = np.random.uniform(0, 1,
                                                                        (N, mask_patchs.shape[3], mask_patchs.shape[4]))
        img2 = np.concatenate(mask_patchs, axis=0, )
        img2 = np.concatenate(img2, axis=1)
        img2 = np.concatenate(img2, axis=1)
        img2 = img2.reshape((N, H, W, C))
        X_mask[:, :ph * patch_size[0], :pw * patch_size[1]] = img2
        return X_mask.swapaxes(1, 3)   # from tf to pytorch


    # initializing weights as 0
    outputs = model(x)
    features = activation["features"].detach()
    weights = torch.zeros_like(features)
    for l in range(ens):
        if l % 1 == 0:
            mask1 = np.random.binomial(1, probb, size=(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]))
            mask2 = np.random.uniform(0, 1, size=(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]))
            mask = np.where(mask1 == 1, 1, mask2)
        elif l % 3 == 1:
            mask = patch_by_strides((batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]), (3, 3), probb)
        elif l % 3 == 2:
            mask = patch_by_strides((batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]), (5, 5), probb)
        else:
            mask = patch_by_strides((batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]), (7, 7), probb)
        mask = torch.tensor(mask, dtype=torch.float32).cuda()
        images_tmp2 = torch.mul(x, mask)
        images_tmp2.requires_grad = True

        logits = model(images_tmp2)
        features = activation["features"]
        label_one_hot = torch.nn.functional.one_hot(y, num_classes).float().cuda().squeeze()
        weights += torch.autograd.grad(torch.mul(logits, label_one_hot).sum(), features)[0].detach()
    weights /= torch.norm(weights, dim=[1,2,3], p=2, keepdim=True)


    adv = x.clone().detach()
    for _ in range(num_iter):
        adv.requires_grad = True
        logits = model(adv)
        features = activation["features"]
        loss = torch.mul(weights, features).sum()
        loss.backward()
        new_grad = adv.grad
        gradients = momentum * gradients + new_grad / torch.norm(new_grad, dim=[1,2,3], p=1, keepdim=True)
        adv = adv - alpha * gradients.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv = torch.max(torch.min(adv, max_x), min_x).detach()
        
    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return x, adv, (pred_top1 != y)



def taig(model, x, y, epsilon=float(8/255), targeted=False):

    def compute_ig(inputs,label_inputs,model):
        device = torch.device('cuda')
        baseline = np.zeros(inputs.shape)
        scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in
                        range(0, steps + 1)]
        scaled_inputs = np.asarray(scaled_inputs)
        if r_flag==True:
            # This is an approximate calculation of TAIG-R
            scaled_inputs = scaled_inputs + np.random.uniform(-epsilon,epsilon,scaled_inputs.shape)
        scaled_inputs = torch.from_numpy(scaled_inputs)
        scaled_inputs = scaled_inputs.to(device, dtype=torch.float)
        scaled_inputs.requires_grad_(True)
        att_out = model(scaled_inputs)
        score = att_out[:, label_inputs]
        loss = -torch.mean(score)
        model.zero_grad()
        loss.backward()
        grads = scaled_inputs.grad.data
        avg_grads = torch.mean(grads, dim=0)
        delta_X = scaled_inputs[-1] - scaled_inputs[0]
        integrated_grad = delta_X * avg_grads
        IG = integrated_grad.cpu().detach().numpy()
        del integrated_grad,delta_X,avg_grads,grads,loss,score,att_out
        return IG

    x, y, model = x.cuda(), y.cuda(), model.cuda().eval()

    niters = 20
    s_num = 20
    r_flag = True

    adv = x.clone()
    adv.requires_grad_(True)
    
    for i in range(niters):
        steps = s_num
        igs = []
        for im_i in range(list(adv.shape)[0]):
            inputs = adv[im_i].cpu().detach().numpy()
            label_inputs = y[im_i]
            integrated_grad = compute_ig(inputs, label_inputs, model)
            igs.append(integrated_grad)
        igs = np.array(igs)

        model.zero_grad()
        input_grad=torch.from_numpy(igs)
        input_grad=input_grad.cuda()

        adv = adv.data + 1./255 * torch.sign(input_grad)
        adv = torch.where(adv > x + epsilon, x + epsilon, adv)
        adv = torch.where(adv < x - epsilon, x - epsilon, adv)
        adv = torch.clamp(adv, min=0, max=1)

    "validate in memory"
    outputs = model(adv)
    pred_top1 = outputs.topk(k=1, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    
    is_adv = (pred_top1 == y) if targeted else (pred_top1 != y)

    return x, adv, is_adv



def evaluate(net, test_set, output_dir=None):
    net.eval().cuda()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=20, shuffle=False,
                                              num_workers=4)
    correct = 0
    i = 0
    for (x, y) in tqdm(test_loader):
        x, y = x.cuda(), y.cuda()
        output = net(x)
        pred_top1 = output.topk(k=1, largest=True).indices
        if pred_top1.dim() >= 2:
            pred_top1 = pred_top1.squeeze()
        correct += (pred_top1 == y).sum().item()
    return correct / len(test_set)

def evaluate_adv(net, test_set, output_dir=None):
    net.eval().cuda()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=20, shuffle=False,
                                              num_workers=4)
    correct = 0
    i = 0
    for (x, y) in tqdm(test_loader):
        x, y = x.cuda(), y.cuda()
        # _, adv, is_adv = clean(net, x, y)
        _, adv, is_adv = fgsm(net, x, y)
        # _, adv, is_adv = pgd(net, x, y)
        # _, adv, is_adv = mi_fgsm(net, x, y)
        # _, adv, is_adv = cw_pgd(net, x, y)
        # _, adv, is_adv = vmi_fgsm(net, x, y)
        # _, adv, is_adv = vmi_ci_fgsm(net, x, y)
        # _, adv, is_adv = fia_attack(net, x, y)
        print("\nattack success rate: {}".format(is_adv.sum().item() / len(is_adv)))
        output = net(adv)
        pred_top1 = output.topk(k=1, largest=True).indices
        if pred_top1.dim() >= 2:
            pred_top1 = pred_top1.squeeze()
        correct += (pred_top1 == y).sum().item()
    return correct / len(test_set)

def main():
    # from vgg import VGG
    # model = VGG("VGG16")
    # model.load_state_dict(torch.load("./vgg_32bit.pth.tar"))
    # normalize = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    # model = nn.Sequential(normalize, model)

    model = torchvision.models.mobilenet_v2(pretrained=True)
    normalize = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    model = nn.Sequential(normalize, model)


    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # test_set = torchvision.datasets.CIFAR10(
    #     root="./data",
    #     train=False,
    #     download=True,
    #     transform=transform,
    # )
    from utils import MyDataset
    test_set = MyDataset(transform=transform)
    clean_acc = evaluate(model, test_set)
    adv_acc = evaluate_adv(model, test_set)
    print("clean_acc: {}".format(clean_acc))
    print("adv_acc: {}".format(adv_acc))
    return


if __name__ == "__main__":
    main()