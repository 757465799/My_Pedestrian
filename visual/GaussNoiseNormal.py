import os
import random
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image


def show_gauss_and_normalized_difference(directory):
    # 列出目录中的所有文件
    all_files = os.listdir(directory)
    # 从文件列表中随机选择一个文件
    random_image_file = random.choice(all_files)
    # 构建完整的文件路径
    image_path = os.path.join(directory, random_image_file)

    # 加载图像
    image = np.array(Image.open(image_path))

    # 定义高斯噪声变换
    gauss_transform = A.Compose([
        A.GaussNoise(var_limit=(1000, 2000), p=1)  # p=1 保证始终应用噪声
    ])

    gauss_blur_transform = A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=1)  # p=1 保证始终应用滤波
    ])

    # 应用高斯滤波变换
    gauss_blur_transformed = gauss_blur_transform(image=image)
    gauss_blur_transformed_image = gauss_blur_transformed["image"]

    # 定义高斯噪声+归一化变换
    gauss_normalize_transform = A.Compose([
        A.GaussNoise(var_limit=(1000, 2000), p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 定义高斯滤波+高斯噪声+归一化变换
    gauss_blur_noise_normalize_transform = A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=1),
        A.GaussNoise(var_limit=(1000, 2000), p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 应用变换
    gauss_transformed = gauss_transform(image=image)
    gauss_transformed_image = gauss_transformed["image"]

    gauss_norm_transformed = gauss_normalize_transform(image=image)
    gauss_norm_transformed_image = gauss_norm_transformed["image"]
    # 计算原始图像与高斯滤波图像之间的差异
    difference_gauss_blur = np.abs(image - gauss_blur_transformed_image)

    # 计算原始图像与变换图像之间的差异
    difference_gauss = np.abs(image - gauss_transformed_image)
    difference_gauss_norm = np.abs(image - gauss_norm_transformed_image)

    # 应用高斯滤波+高斯噪声+归一化变换
    gauss_blur_noise_norm_transformed = gauss_blur_noise_normalize_transform(image=image)
    gauss_blur_noise_norm_transformed_image = gauss_blur_noise_norm_transformed["image"]

    # 计算原始图像与高斯滤波+高斯噪声+归一化图像之间的差异
    difference_gauss_blur_noise_norm = np.abs(image - gauss_blur_noise_norm_transformed_image)

    # 展示结果
    fig, ax = plt.subplots(1, 9, figsize=(32, 5))  # 修改为 9 个子图

    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title(f"Original: {random_image_file}")

    ax[1].imshow(gauss_blur_transformed_image)
    ax[1].axis('off')
    ax[1].set_title("Gaussian Blur")

    ax[2].imshow(difference_gauss_blur)
    ax[2].axis('off')
    ax[2].set_title("Difference with Blur")

    ax[3].imshow(gauss_transformed_image)
    ax[3].axis('off')
    ax[3].set_title("Gauss Noise")

    ax[4].imshow(difference_gauss)
    ax[4].axis('off')
    ax[4].set_title("Difference with Gauss")

    ax[5].imshow(gauss_norm_transformed_image, cmap='gray')
    ax[5].axis('off')
    ax[5].set_title("Gauss + Normalize")

    ax[6].imshow(difference_gauss_norm, cmap='gray')
    ax[6].axis('off')
    ax[6].set_title("Difference with Gauss + Norm")

    ax[7].imshow(gauss_blur_noise_norm_transformed_image, cmap='gray')
    ax[7].axis('off')
    ax[7].set_title("Blur + Gauss + Norm")

    ax[8].imshow(difference_gauss_blur_noise_norm, cmap='gray')
    ax[8].axis('off')
    ax[8].set_title("Difference with Blur + Gauss + Norm")

    plt.tight_layout()
    plt.show()


# 使用函数
show_gauss_and_normalized_difference("C:/Users/h/pythonProject/LabelmeToCoco/annotations/train/JPEGImages/")
