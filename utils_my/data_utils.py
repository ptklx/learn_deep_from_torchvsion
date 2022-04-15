# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : data_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import json
import random

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
import math

from typing import Tuple

# import torch
from torch import Tensor
from torchvision.transforms import functional as F
import torch.distributed as dist

def read_split_data(root: str, val_rate: float = 0.2, plot_image: bool = False):
    # 保证随机结果可复现
    random.seed(0)
    assert os.path.exists(root), f'dataset root {root} does not exist.'

    # 遍历文件夹，一个文件夹对应一个类别
    flower_classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    # 排序，保证顺序一致
    flower_classes.sort()

    # 给类别进行编码，生成对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as f:
        f.write(json_str)

    # 训练集所有图片的路径和对应索引信息
    train_images_path, train_images_label = [], []

    # 验证集所有图片的路径和对应索引信息
    val_images_path, val_images_label = [], []

    # 每个类别的样本总数
    every_class_num = []

    # 支持的图片格式
    images_format = [".jpg", ".JPG", ".png", ".PNG"]

    # 遍历每个文件夹下的文件
    for cla in flower_classes:
        cla_path = os.path.join(root, cla)

        # 获取每个类别文件夹下所有图片的路径
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in images_format]

        # 获取类别对应的索引
        image_class = class_indices[cla]

        # 获取此类别的样本数
        every_class_num.append(len(images))

        # 按比例随机采样验证集
        # val_path = random.sample(images, k=int(len(images) * val_rate))
        # for img_path in images:
        #     if img_path in val_path:
        #         val_images_path.append(img_path)
        #         val_images_label.append(image_class)
        #     else:
        #         train_images_path.append(img_path)
        #         train_images_label.append(image_class)

        ####对于数据比较少时,不用测试数据,训练数据就是测试数据##
        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)
            train_images_path.append(img_path)
            train_images_label.append(image_class)




    print(f"{sum(every_class_num)} images found in dataset.")
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")

    if plot_image:
        plt.bar(range(len(flower_classes)), every_class_num, align='center')
        plt.xticks(range(len(flower_classes)), flower_classes)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label ,len(flower_classes)


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_label: list, transform=None,classes=1):
        self.images_path = images_path
        self.images_label = images_label
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError(f"image: {self.images_path[item]} is not RGB mode")
        label = self.images_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def get_dataset_dataloader(data_path):
    train_images_path, train_iamges_label, val_images_path, val_images_label,f_classes = read_split_data(root=data_path)

    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                 transforms.RandomHorizontalFlip(),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),

    #     "val": transforms.Compose([transforms.Resize(224),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # }

    data_transform = {
        "train": transforms.Compose([
                                    # transforms.Resize([224,224]),
                                    transforms.RandomResizedCrop([224,224], scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    # scale是指相对于原图的擦除面积范围
                                    # ratio是指擦除区域的宽高比
                                    # value是指擦除区域的值，如果是int，也可以是tuple（RGB3个通道值），或者是str，需为'random'，表示随机生成
                                    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                                    
                                    
                                    ]),
                                    

        "val": transforms.Compose([
                                #    transforms.Resize([224,224]),
                                   transforms.Resize([256,256]),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_label=train_iamges_label,
                              transform=data_transform['train'],classes=f_classes)
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_label=val_images_label,
                            transform=data_transform['val'],classes=f_classes)

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # print(f"Using {nw} dataloader workers every process.")

    # train_dataloader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=nw,
    #     collate_fn=train_dataset.collate_fn
    # )
    # val_dataloader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=nw,
    #     collate_fn=val_dataset.collate_fn
    # )

    # return train_dataset, val_dataset, train_dataloader, val_dataloader
    return train_dataset, val_dataset

# from https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0. # beta分布超参数
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
  
        # 建立one-hot标签
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)
  
        # 判断是否进行mixup
        if torch.rand(1).item() >= self.p:
            return batch, target
  
        # 这里将batch数据平移一个单位，产生mixup的图像对，这意味着每个图像与相邻的下一个图像进行mixup
        # timm实现是通过flip来做的，这意味着第一个图像和最后一个图像进行mixup
        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
  
        # 随机生成组合系数
        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled) # 得到mixup后的图像

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled) # 得到mixup后的标签

        return batch, target

class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = F.get_image_size(batch)
  
        # 确定patch的起点
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))
  
        # 确定patch的w和h（其实是一半大小）
        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)
  
        # 越界处理
        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        # 由于越界处理， λ可能发生改变，所以要重新计算
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # 重复采样后每个replica的样本量
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        # 重复采样后的总样本量
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        # 每个replica实际样本量，即不重复采样时的每个replica的样本量
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)] # 重复3次
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample: 使得同一个样本的重复版本进入不同的进程（GPU）
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples]) # 截取实际样本量

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch