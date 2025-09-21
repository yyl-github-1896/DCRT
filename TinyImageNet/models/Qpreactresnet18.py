import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.ao.quantization import default_qconfig, prepare, convert
from tqdm import tqdm

device = "cpu"

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # print(out.dtype)
        out = torch.add(out, shortcut) # 修改为量化友好的加法
        return out

class QPreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(QPreActResNet, self).__init__()
        self.in_planes = 64
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.dequant = torch.quantization.DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = self.conv1(x)
        print('After conv1:', out.min().item(), out.max().item())
        out = self.layer1(out)
        print('After layer1:', out.min().item(), out.max().item())
        out = self.layer2(out)
        print('After layer2:', out.min().item(), out.max().item())
        out = self.layer3(out)
        print('After layer3:', out.min().item(), out.max().item())
        out = self.layer4(out)
        print('After layer4:', out.min().item(), out.max().item())
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.dequant(out)
        print('After dequant:', out.min().item(), out.max().item())
        return out


def Qpreactresnet18():
    return QPreActResNet(PreActBlock, [2,2,2,2])

