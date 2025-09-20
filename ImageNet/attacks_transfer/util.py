import numpy as np
import imageio
from PIL import Image
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = imageio.imread(filepath)  # 使用 imageio 替代 imread
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)  # 如果图像只有一个通道，扩展为三个通道
        img = np.repeat(img, 3, axis=2)  # 复制单通道成三个通道（灰度转RGB）
    
    # 使用 PIL 进行图像大小调整
    img = Image.fromarray(img)
    img = img.resize((256, 256))  # 调整为 256x256
    img = np.array(img)
    
    img = np.transpose(img, (2, 0, 1))  # 将图像转换为 (C, H, W) 格式
    img = torch.from_numpy(img)
    img = preprocess_img(img)
    return img


def save_img(img, filename):
    img = deprocess_img(img)
    img = img.numpy()
    img *= 255.0  # 将像素值还原为 [0, 255] 范围
    img = img.clip(0, 255)  # 截取超出范围的像素值
    img = np.transpose(img, (1, 2, 0))  # 转换回 (H, W, C) 格式

    # 使用 PIL 来保存图像
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((250, 200))  # 保存时调整为 250x200
    img.save(filename)
    print(f"Image saved as {filename}")


def preprocess_img(img):
    # 将 [0, 255] 范围的图像转换为 [0, 1]
    min = img.min()
    max = img.max()
    img = torch.FloatTensor(img.size()).copy_(img)
    img.add_(-min).mul_(1.0 / (max - min))

    # 将 RGB 转换为 BGR
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    # 将 [0, 1] 范围的图像转换为 [-1, 1]
    img = img.mul_(2).add_(-1)

    # 检查输入图像是否在预期范围内
    assert img.max() <= 1, 'badly scaled inputs'
    assert img.min() >= -1, "badly scaled inputs"

    return img


def deprocess_img(img):
    # 将 BGR 转换回 RGB
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    # 将 [-1, 1] 范围的图像转换为 [0, 1]
    img = img.add_(1).div_(2)

    return img
