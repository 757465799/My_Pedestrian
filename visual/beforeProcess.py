import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from Dataset_Process import CustomDataset


def show_image_and_annotations(image, boxes, labels):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(image)
    for box, label in zip(boxes, labels):
        # 注意：这里我们假设box是COCO格式（[x, y, width, height]）
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, str(label), fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()


def show_random_sample_from_dataset(dataset):
    idx = np.random.randint(0, len(dataset))
    image_path = os.path.join(dataset.images_dir, dataset.image_file_names[idx])
    image = Image.open(image_path).convert("RGB")

    image_id = dataset.image_ids[idx]
    annotations = [ann for ann in dataset.coco_data['annotations'] if ann['image_id'] == image_id]
    boxes = [ann['bbox'] for ann in annotations]
    labels = [ann['category_id'] for ann in annotations]

    show_image_and_annotations(image, boxes, labels)


# 使用函数展示一张随机样本
dataset = CustomDataset("C:/Users/h/pythonProject/LabelmeToCoco/annotations/train", "C:/Users/h/pythonProject/LabelmeToCoco/annotations/train/instances_train.json")
show_random_sample_from_dataset(dataset)
