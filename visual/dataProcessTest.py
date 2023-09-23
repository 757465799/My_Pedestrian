import matplotlib.pyplot as plt
import numpy as np
import os
import json
import torch
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from Dataset_Process import load_dataset


# 请确保您已经定义了load_dataset和CustomDataset

def show_random_sample(train_loader):
    # 从数据加载器中随机选择一个批次的数据
    images, targets = next(iter(train_loader))

    # 从该批次中随机选择一张图片及其相关标注
    idx = np.random.randint(0, len(images))
    image = images[idx].permute(1, 2, 0).numpy()
    # 将图像从归一化状态恢复
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    target = targets[idx]
    image_width = 1280
    image_height = 720
    # 展示图片及其标注
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(image)
    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()
    for box, label in zip(boxes, labels):
        x, y, w, h = box
        x, y = x - w / 2, y - h / 2  # 将框从中心点-宽高格式转回到左上角-宽高格式
        rect = plt.Rectangle((x * image_width, y * image_height), w * image_width, h * image_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x * 512, y * 512, str(label), fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()


# 加载数据
train_loader, _, _ = load_dataset("C:/Users/h/pythonProject/LabelmeToCoco/annotations")
show_random_sample(train_loader)
