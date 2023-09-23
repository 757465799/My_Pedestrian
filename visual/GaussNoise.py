import os
import random
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image


def show_gauss_noise_and_difference_from_dir(directory):
    # 列出目录中的所有文件
    all_files = os.listdir(directory)
    # 从文件列表中随机选择一个文件
    random_image_file = random.choice(all_files)
    # 构建完整的文件路径
    image_path = os.path.join(directory, random_image_file)

    # 加载图像
    image = np.array(Image.open(image_path))

    # 定义只有高斯噪声的变换
    gauss_transform = A.Compose([
        A.GaussNoise(var_limit=(1000, 2000), p=1)  # p=1 保证始终应用噪声
    ])

    # 应用变换
    transformed = gauss_transform(image=image)
    transformed_image = transformed["image"]

    # 计算差异图像
    difference = np.abs(image - transformed_image)
    mse = np.mean((image - transformed_image) ** 2)

    # 展示结果
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title(f"Original: {random_image_file}")

    ax[1].imshow(transformed_image)
    ax[1].axis('off')
    ax[1].set_title("Transformed")

    ax[2].imshow(difference)
    ax[2].axis('off')
    ax[2].set_title(f"Difference (MSE: {mse:.2f})")

    plt.tight_layout()
    plt.show()


# 使用函数
show_gauss_noise_and_difference_from_dir("C:/Users/h/pythonProject/LabelmeToCoco/annotations/train/JPEGImages/")

