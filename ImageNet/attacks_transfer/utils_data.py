import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import json
import pickle
import pandas as pd

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

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

class ImagePathDataset(VisionDataset):
    def __init__(self, config, transform=None, target_transform=None,
                 loader=default_loader, return_paths=False):
        super().__init__(root=config["root"], transform=transform, target_transform=target_transform)
        self.config = config

        self.loader = loader
        self.extensions = IMG_EXTENSIONS

        self.classes = config["classes"]
        self.class_to_idx = config["class_to_idx"]
        self.samples = config["samples"]
        self.targets = [s[1] for s in self.samples]
        self.return_paths = return_paths

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = sample, target

        if self.return_paths:
            return output, path
        else:
            return output

    def __len__(self):
        return len(self.samples)

    @classmethod
    def from_path(cls, config_path, *args, **kwargs):
        with open(config_path, mode="r") as f:
            return cls(config=json.loads(f.read()), *args, **kwargs)

    
def set_requires_grad(named_parameters, requires_grad):
    for name, param in named_parameters:
        param.requires_grad = requires_grad



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

def generate_data_pickle():
    all_image = dict()

    # 根据 CSV 文件构造标签到图片路径的映射（假设每个类别只有一张图片）
    import pandas as pd
    csv_path = '/data/crq/data/val_rs.csv'
    df = pd.read_csv(csv_path)
    
    # 这里构造一个字典，key 为类别标签，value 为图片完整路径
    # 注意：若 CSV 中标签为字符串，需要根据实际情况转换成数字
    label_to_filepath = {int(row['label']): os.path.join('/data/crq/data/val_clean', row['filename']) for idx, row in df.iterrows()}
    
    for label, filepath in label_to_filepath.items():
        try:
            obj = Image.open(filepath)
            obj = obj.convert('RGB')
            all_image[label] = [obj]  # 每个类别只有一张图片
        except Exception as e:
            print(f"无法打开 {filepath}: {e}")
    
    with open('data.pickle', 'wb') as f:
        pickle.dump(all_image, f, pickle.HIGHEST_PROTOCOL)

