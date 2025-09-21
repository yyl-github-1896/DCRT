import argparse
import copy
import models
import dataset_configs
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
import ops_attack
from utils2 import *
from utils2 import cprint
from torch.autograd import Variable
from tqdm import tqdm
from argparse import Namespace
# import improved_diffusion
parser = argparse.ArgumentParser(description='generate dataset for data-centric robust learning')
parser.add_argument('--id', default='0112', type=str, help='experiment id')
parser.add_argument('--batch_size', default=256, type=int, help='')
parser.add_argument('--data', default='/data/crq/data', type=str, help='path to cifar dataset')
parser.add_argument('--device', default='0, 1, 2, 3, 4, 5', type=str, help='gpu device')
parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()
def tensor2np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((0, 2, 3, 1))
    return img
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
        assert index < self.__len__()
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
def split_dataset(batch_size, output_dir='./datasets/split/', source='merged'):
    '''split merged data into 5 balanced subsets'''
    sub_imgs = np.zeros((5, 20000, 32, 32, 3), dtype=np.uint8)  
    sub_labels = np.zeros((5, 20000, 10), dtype=np.float32)

    if source == 'merged':
        print('Loading merged 100K dataset...')
        merged_images = np.load('/data/crq/DRL/cifar100/DCRT/merge/images.npy')  
        merged_labels = np.load('/data/crq/DRL/cifar100/DCRT/merge/labels.npy')  

        class_indices = {}
        for class_id in range(10):
            mask = np.argmax(merged_labels, axis=1) == class_id
            class_indices[class_id] = np.where(mask)[0]
            assert len(class_indices[class_id]) == 10000, f"Class {class_id} imbalance!"

        for class_id in range(10):

            np.random.shuffle(class_indices[class_id])
            
            for sub_id in range(5):
                start = sub_id * 2000
                end = (sub_id + 1) * 2000
                subset_indices = class_indices[class_id][start:end]

                sub_start = class_id * 2000
                sub_end = (class_id + 1) * 2000
                
                sub_imgs[sub_id, sub_start:sub_end] = merged_images[subset_indices]
                sub_labels[sub_id, sub_start:sub_end] = merged_labels[subset_indices]

    for idx in range(5):
        img_dest = os.path.join(output_dir, f"images{idx}.npy")
        label_dest = os.path.join(output_dir, f"labels{idx}.npy")
        os.makedirs(output_dir, exist_ok=True)
        np.save(img_dest, sub_imgs[idx])
        np.save(label_dest, sub_labels[idx])
    
    return sub_imgs, sub_labels

def main():
    sub_clean_imgs, sub_labels = split_dataset(100)
    sub_adv_imgs = copy.deepcopy(sub_clean_imgs)
    dataset_args = dataset_configs.__dict__['args_' + args.id]
    for dataset_id, (k, v) in enumerate(dataset_args.items()):
            print('generating dataset {}...'.format(dataset_id))
            attacker = ops_attack.__dict__[v['attack']]
            source_args = v['source_model']
            model_list = []
            for model_arg in source_args:
                from res50 import resnet50
                model = resnet50()
                model.load_state_dict(torch.load('/data/crq/DRL/ckpt/dcrt/resnet50_cifar1000.pth'))
                normalize = ops_attack.NormalizeByChannelMeanStd(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
                model = torch.nn.Sequential(normalize, model).cuda().eval()
                model_list.append(model)
            source_model = model
            subset = MyDataset(transform=transforms.Compose([transforms.ToTensor(),]), img_path='./datasets/split/images{}.npy'.format(dataset_id), label_path='./datasets/split/labels{}.npy'.format(dataset_id))
            loader = torch.utils.data.DataLoader(subset, batch_size=1000, shuffle=False,
                                                num_workers=4)
            from models.generators import ResnetGenerator, weights_init
            netG = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=[0])
            checkpoint = '/data/crq/DRL/cifar100/attacks_transfer/gapmodel/netG_model_epoch_9_'
            netG.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
            for img_id, (x, y) in enumerate(loader):
                x, y = x.cuda(), y.cuda()
                target = y.topk(k=1, largest=True).indices.squeeze()
                from ops_attack import mi_fgsm,ni_fgsm,pi_fgsm,di_fgsm,admix,ssa,ILA,fia,NAA,GAP,GAPF,BIA,IAA,SGM,RFA
                if attacker is fia:
                    resnet50_model = source_model._modules.get('1')
                    source_layers = list(enumerate(map(lambda name: (name, resnet50_model._modules.get(name)), ['conv1', 'layer1', 'layer2','layer3','layer4', 'fc'])))
                    _, adv, is_adv = attacker(source_model, x, y, source_layers[2][1][1], learning_rate=8/2550, epsilon=8/255, niters=10, dataset='cifar100')
                elif attacker is GAP or attacker is GAPF or attacker is BIA:
                    _, adv, is_adv = attacker(source_model,netG, x, target, epsilon=float(8/255))
                elif attacker is ILA:
                    from utils import ifgsm
                    ifgsm_guide = ifgsm(source_model, x, y, learning_rate=8/2550, epsilon=8/255, niters=10, dataset='cifar100')
                    source_layers=list(enumerate(map(lambda name: (name, source_model._modules.get(name)), ['conv1', 'layer1', 'layer2','layer3','layer4', 'fc'])))
                    _, adv, is_adv = attacker(source_model, x, ifgsm_guide, y, source_layers[2][1][1],learning_rate=8/2550, epsilon=8/255, niters=10, dataset='cifar100')
                elif attacker is NAA:
                    source_layers=list(enumerate(map(lambda name: (name, source_model._modules.get(name)), ['conv1', 'layer1', 'layer2','layer3','layer4', 'fc'])))
                    _, adv, is_adv = attacker(source_model, x, y, source_layers[2][1][1],learning_rate=8/2550, epsilon=8/255, niters=10, dataset='cifar100')
                else:
                    _, adv, is_adv = attacker(source_model, x, target, epsilon=float(8/255))
                print("\nattack success rate: {}".format(is_adv.sum().item() / len(is_adv)))
                sub_adv_imgs[dataset_id][img_id * 1000: img_id * 1000 + len(y)] = tensor2np(adv).astype(np.uint8)

            '''label smoothing'''
            for label_id in range(len(sub_labels[dataset_id])):
                y = np.argmax(sub_labels[dataset_id][label_id])
                sub_labels[dataset_id][label_id] = np.ones(10, dtype=np.float32) * ((1 - v['label_smoothing']) / 9)
                sub_labels[dataset_id][label_id][y] = v['label_smoothing']



    '''concacte'''
    print('Concacting and sampling 50K...')

    final_images = np.zeros((2, 50000, 32, 32, 3), dtype=np.uint8)
    final_labels = np.zeros((50000, 10), dtype=np.float32)
    
    subset_ratios = {
        0: 0.2,
        1: 0.2, 
        2: 0.2, 
        3: 0.2, 
        4: 0.2  
    }

    for subset_id in range(5):
        original_count = 20000 
        sample_count = int(50000 * subset_ratios[subset_id])

        class_samples = sample_count // 10 
        indices = []
        for class_id in range(10):

            class_mask = np.argmax(sub_labels[subset_id], axis=1) == class_id
            class_indices = np.where(class_mask)[0]

            selected = np.random.choice(
                class_indices, 
                size=class_samples, 
                replace=False
            )
            indices.extend(selected)
        start = subset_id * 10000 
        end = start + sample_count

        np.random.shuffle(indices)

        final_images[0, start:end] = sub_clean_imgs[subset_id][indices]
        final_images[1, start:end] = sub_adv_imgs[subset_id][indices]
        final_labels[start:end] = sub_labels[subset_id][indices]

    final_images = final_images.swapaxes(0, 1) 

    save_root_path = f'./datasets/dcrt'
    os.makedirs(save_root_path, exist_ok=True)
    
    np.save(os.path.join(save_root_path, 'images.npy'), final_images)
    np.save(os.path.join(save_root_path, 'labels.npy'), final_labels)

if __name__ == "__main__":
    main()
