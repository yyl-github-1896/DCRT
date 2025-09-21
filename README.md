# DCRT

This repository contains code to reproduce results from the paper:

[Data-Centric Robust Training for Defending against Transfer-based Adversarial Attacks](https://ieeexplore.ieee.org/document/11168936/) (TIFS 2025)

Transfer-based adversarial attacks pose a severe threat to real-world deep learning systems since they do not require access to target models. Adversarial training (AT), which is recognized as the most effective defense against white-box attacks, also ensures high robustness against (black-box) transfer-based attacks. However, AT suffers from significant computational overhead because it repeatedly generates adversarial examples (AEs) throughout the entire training process. In this paper, we demonstrate that such repeated generation is unnecessary to achieve robustness against transfer-based attacks. Instead, pre-generating AEs all at once before training is sufficient, as proposed in our new defense paradigm called Data-Centric Robust Training (DCRT). DCRT employs clean data augmentation and adversarial data augmentation techniques to enhance the dataset before training. Our experimental results show that DCRT outperforms widely-used AT techniques (e.g., PGD-AT, TRADES, EAT, and FAT) in terms of transfer-based black-box robustness and even surpasses the top-1 defense on RobustBench when combined with common model-centric techniques. We also highlight additional benefits of DCRT, such as improved training efficiency and class-wise fairness.

![DCRT.png](https://github.com/coolbomb1/DCRT/blob/main/DCRT.png)

## Requirements

+ Python >= 3.8.0
+ torch >= 1.10.2
+ torchvision >= 0.11.3
+ numpy >= 1.20.3
+ pandas >= 1.4.1
+ PIL >= 9.3.0
+ robustbench
+ tqdm >= 4.62.3



## CIFAR-10

### Prepare the data

The CIFAR-10 dataset will be downloaded automatically when running the code. 

### Running the DCRT method on CIFAR-10

you can run the following script:
```
bash ./CIFAR10/DCRT_CIFAR10.sh
```
you could also download our [pre-trained DCRT models](https://drive.google.com/file/d/1KmM5sEnJo-Y5B-KMhIjINcM13WEDD5Ls/view?usp=sharing) for CIFAR10.

## CIFAR-100

### Prepare the data and models

The CIFAR-100 dataset will be downloaded automatically when running the code.

### Running the DCRT method on CIFAR-100

you can run the following script:
```
bash ./CIFAR100/DCRT_CIFAR100.sh
```
you could also download our [pre-trained DCRT models](https://drive.google.com/file/d/1X-FW0ExHDZ1C8pwP1T1aNH-BAJfwENlN/view?usp=sharing) for CIFAR100.

## TinyImageNet

### Prepare the data and models

The TinyImageNet dataset could be downloaded from [data](http://cs231n.stanford.edu/tiny-imagenet-200.zip). The directory structure should be like:

```
tiny-imagenet-200
+-- train
+-- val
+-- test
```

### Running the DCRT method on TinyImageNet

you can run the following script:
```
bash ./TinyImageNet/DCRT_TinyImageNet.sh
```
you could also download our [pre-trained DCRT models](https://drive.google.com/file/d/1vdVSnqSDHTmK6EUE__4ftENgI-76vXAV/view?usp=sharing) for TinyImageNet.

## ImageNet

### Prepare the data and models

The ImageNet dataset could be downloaded from [data](https://www.image-net.org/). The directory structure should be like:

```
ImageNet
+-- train
+-- val
```

### Running the DCRT method on ImageNet

you can run the following script:
```
bash ./ImageNet/DCRT_ImageNet.sh
```
you could also download our [pre-trained DCRT models](https://drive.google.com/file/d/1LnGwP-_XZKOOPTGhlk02PEIBk8NJ90T6/view?usp=sharing) for ImageNet.

## About us
We are in XJTU-AISEC lab led by [Prof. Chao Shen](https://gr.xjtu.edu.cn/en/web/cshen/home), [Prof. Chenhao Lin](https://gr.xjtu.edu.cn/en/web/linchenhao), [Prof. Zhengyu Zhao](https://zhengyuzhao.github.io/), Prof. Qian Li, and etc. in the School of Cyber Science and Engineering, Xi'an Jiaotong University.

Please contact Yulong Yang, Xiang Ji and Ruiqi Cao at xjtu2018yyl0808@stu.xjtu.edu.cn, xiangji@stu.xjtu.edu.cn and crq2002@stu.xjtu.edu.cn if you have any question on the codes. If you find this repository useful, please consider giving ⭐.
