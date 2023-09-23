import os
import random
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2


def show_all_transformed_image(directory):
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
        A.GaussNoise(var_limit=(1000, 2000), p=1)
    ])

    # 定义高斯噪声+归一化变换
    gauss_normalize_transform = A.Compose([
        A.GaussNoise(var_limit=(1000, 2000), p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 定义所有变换
    all_transforms = A.Compose([
        A.GaussNoise(var_limit=(1000, 2000), p=1),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 应用变换
    gauss_transformed = gauss_transform(image=image)
    gauss_transformed_image = gauss_transformed["image"]

    gauss_norm_transformed = gauss_normalize_transform(image=image)
    gauss_norm_transformed_image = gauss_norm_transformed["image"]

    all_transformed = all_transforms(image=image)
    all_transformed_image = all_transformed["image"].permute(1, 2, 0).numpy()

    # 展示结果
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title(f"Original: {random_image_file}")

    ax[1].imshow(gauss_transformed_image)
    ax[1].axis('off')
    ax[1].set_title("Gauss Noise")

    ax[2].imshow(gauss_norm_transformed_image, cmap='gray')
    ax[2].axis('off')
    ax[2].set_title("Gauss + Normalize")

    ax[3].imshow(all_transformed_image, cmap='gray')
    ax[3].axis('off')
    ax[3].set_title("All Transforms (incl. ToTensor)")

    plt.tight_layout()
    plt.show()


# 使用函数
show_all_transformed_image("C:/Users/h/pythonProject/LabelmeToCoco/annotations/train/JPEGImages/")
