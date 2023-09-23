# import os
# from pathlib import Path
# import json
#
# import numpy as np
# import torch
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image

import os
import json

import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_dataset(coco_root):
    # 定义文件和目录的路径
    train_images_dir = os.path.join(coco_root, "train")
    val_images_dir = os.path.join(coco_root, "val")
    test_images_dir = os.path.join(coco_root, "test")
    train_json_path = os.path.join(coco_root, "train", "instances_train.json")
    val_json_path = os.path.join(coco_root, "val", "instances_val.json")
    test_json_path = os.path.join(coco_root, "test", "instances_test.json")

    train_transform = get_train_transform()
    val_test_transform = get_val_test_transform()

    train_dataset = CustomDataset(train_images_dir, train_json_path, transform=train_transform)
    val_dataset = CustomDataset(val_images_dir, val_json_path, transform=val_test_transform)
    test_dataset = CustomDataset(test_images_dir, test_json_path, transform=val_test_transform)

    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # 将所有图像堆叠到一个张量中
        images = torch.stack(images, dim=0)

        # 在目标中，我们有变长的boxes和labels，因此我们只能将它们放入列表中
        # 这是因为不是所有的图像都有相同数量的目标
        return images, targets

    batch_size = 4
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def get_train_transform():
    bbox_params = A.BboxParams(
        format='coco',
        label_fields=['labels']
    )

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(var_limit=(200, 500), p=0.5),
        # A.GaussianBlur(p=0.5),  # 添加高斯滤波
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=bbox_params)

    return train_transform


def get_val_test_transform():
    bbox_params = A.BboxParams(
        format='coco',
        label_fields=['labels']
    )

    val_test_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=bbox_params)

    return val_test_transform



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # 加载COCO JSON文件
        with open(json_path) as f:
            self.coco_data = json.load(f)

        self.image_ids = [img["id"] for img in self.coco_data["images"]]
        self.image_file_names = [img["file_name"] for img in self.coco_data["images"]]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.images_dir, self.image_file_names[index])

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 获取边界框和标签
        annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]
        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]

        if self.transform:
            transformed = self.transform(image=np.array(image), bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]

        # print("After transformation:", boxes)
        # COCO 格式的边界框是[x, y, width, height]
        # 转换 boxes 到中心点-宽高格式
        boxes = [[box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3]] for box in boxes]
        # print("After transformation:", boxes)
        image_width = 1280
        image_height = 720
        boxes = [[box[0] / image_width, box[1] / image_height, box[2] / image_width, box[3] / image_height] for box in
                 boxes]
        # print("Before transformation:", boxes)
        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32)
        }

        return image, target

    def __len__(self):
        return len(self.image_file_names)


'''

            TEST
            测试

'''


# def visualize_random_preprocessed_sample_from_path():
#     train_images_dir = 'C:/Users/h/pythonProject/LabelmeToCoco/annotations/train/'
#     train_json_path = 'C:/Users/h/pythonProject/LabelmeToCoco/annotations/train/instances_train.json'
#
#     bbox_params = A.BboxParams(
#         format='coco',  # 边界框的格式[x, y, width, height]
#         label_fields=['labels']
#     )
#     # train_transform = A.Compose([
#     #     # 数据增强策略
#     #     # A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
#     #     A.HorizontalFlip(p=0.5),
#     #     # A.RandomScale(scale_limit=0.1, p=0.5),
#     #     A.GaussNoise(var_limit=(10, 50), p=0.5),
#     #     # A.MotionBlur(p=0.2),
#     #     # A.GaussianBlur(p=0.2),
#     #
#     #     A.Resize(256, 256),
#     #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     #     ToTensorV2()
#     # ], bbox_params=bbox_params)
#
#     dataset = CustomDataset(train_images_dir, train_json_path)
#
#     index = np.random.randint(0, len(dataset))
#
#     image, target = dataset[index]
#     # Convert image tensor back to PIL image for visualization
#     image = image.permute(1, 2, 0).numpy()
#     image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
#     image = (image * 255).astype(np.uint8)
#
#     fig, ax = plt.subplots(1, figsize=(12, 9))
#     ax.imshow(image)
#
#     boxes = target['boxes'].numpy()
#     labels = target['labels'].numpy()
#
#     for box, label in zip(boxes, labels):
#         # Convert center-width format back to top-left format for visualization
#         x, y, width, height = box
#         x, y = x - width / 2, y - height / 2
#
#         rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         plt.text(x, y, str(label), color="r")
#
#     plt.axis('off')
#     plt.show()
#
#
# # Usage:
# visualize_random_preprocessed_sample_from_path()

# dataset_path = "C:/Users/h/pythonProject/LabelmeToCoco/"
#
# # 拆分数据集
# train_loader, val_loader = load_dataset(dataset_path)
#
# # 从数据加载器中随机抽取一个样本
# images, target_list = next(iter(train_loader))
# sample_image = images[0].numpy().transpose(1, 2, 0)
# sample_target = target_list[0]  # Select the target for the first image
# sample_bbox = sample_target['boxes'].numpy()
#
# # 逆标准化图像
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# sample_image = sample_image * std + mean
# sample_image = np.clip(sample_image, 0, 1)
#
# # 使用原始图像的尺寸进行反归一化
# original_height, original_width = 720, 1280
# sample_bbox = np.array(sample_bbox)
# if len(sample_bbox.shape) == 1:
#     sample_bbox = sample_bbox.reshape(1, -1)
#
#
# def center_width_height_to_voc(boxes):
#     """
#     将中心点-宽高格式的边界框转换为Pascal VOC格式
#     boxes: (center_x, center_y, width, height)
#     return: (x_min, y_min, x_max, y_max)
#     """
#     center_x, center_y, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#     x_min = center_x - (width / 2)
#     y_min = center_y - (height / 2)
#     x_max = center_x + (width / 2)
#     y_max = center_y + (height / 2)
#     return np.stack([x_min, y_min, x_max, y_max], axis=1)
#
#
# # 转换边界框格式
# sample_bbox = center_width_height_to_voc(sample_bbox)
#
# # 使用原始图像的尺寸对归一化的边界框坐标进行反归一化
# sample_bbox[:, [0, 2]] *= original_width
# sample_bbox[:, [1, 3]] *= original_height
#
# # 使用matplotlib来展示图像和边界框
# fig, ax = plt.subplots(1)
# ax.imshow(sample_image)
#
# for box in sample_bbox:
#     x_min, y_min, x_max, y_max = box
#     rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
#
# plt.show()

# def load_dataset(dataset_path, test_size=0.1, val_size=0.2):
#     # 加载图片路径和标签
#     image_paths = []
#     labels = []
#     for root, dirs, files in os.walk(dataset_path):
#         for file in files:
#             if file.endswith(".jpg"):
#                 image_path = os.path.join(root, file)
#                 label_path = os.path.join(root, file.replace(".jpg", ".json"))
#                 image_paths.append(image_path)
#                 labels.append(label_path)
#
#     # 拆分数据集
#     train_val_images, test_images, train_val_labels, test_labels = train_test_split(
#         image_paths, labels, test_size=test_size, random_state=42)
#     train_images, val_images, train_labels, val_labels = train_test_split(
#         train_val_images, train_val_labels, test_size=val_size, random_state=42)
#
#     # 定义bbox_params
#     bbox_params = A.BboxParams(
#         format='pascal_voc',  # 边界框的格式[x_min, y_min, x_max, y_max]
#         label_fields=['labels']
#     )
#
#     train_transform = A.Compose([
#         # 数据增强策略
#         A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
#         A.HorizontalFlip(p=0.5),
#         # A.RandomScale(scale_limit=0.1, p=0.5),
#         A.GaussNoise(var_limit=(10, 50), p=0.5),
#         # A.MotionBlur(p=0.2),
#         # A.GaussianBlur(p=0.2),
#
#         A.Resize(256, 256),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ], bbox_params=bbox_params)
#
#     # 对于验证和测试数据
#     val_test_transform = A.Compose([
#         A.Resize(256, 256),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ], bbox_params=bbox_params)
#
#     train_dataset = CustomDataset(train_images, train_labels, transform=train_transform, )
#     val_dataset = CustomDataset(val_images, val_labels, transform=val_test_transform)
#     test_dataset = CustomDataset(test_images, test_labels, transform=val_test_transform)
#
#     def collate_fn(batch):
#         images = [item[0] for item in batch]
#         targets = [item[1] for item in batch]
#
#         # 将所有图像堆叠到一个张量中
#         images = torch.stack(images, dim=0)
#
#         # 在目标中，我们有变长的boxes和labels，因此我们只能将它们放入列表中
#         # 这是因为不是所有的图像都有相同数量的目标
#         return images, targets
#
#     batch_size = 8
#     # 创建数据加载器
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
#     return train_loader, val_loader, test_loader
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, image_paths, label_paths, transform=None):
#         self.image_paths = image_paths
#         self.label_paths = label_paths
#         self.transform = transform
#
#     def __getitem__(self, index):
#         image_path = self.image_paths[index]
#         label_path = self.label_paths[index]
#
#         # 加载图像
#         image = Image.open(image_path).convert("RGB")
#
#         # 加载标签
#         with open(label_path) as f:
#             label_data = json.load(f)
#
#         # 获取边界框并归一化
#         boxes = []
#         width = label_data["imageWidth"]
#         height = label_data["imageHeight"]
#         for item in label_data["shapes"]:
#             if item["shape_type"] == "rectangle":
#                 box = [coord for point in item["points"] for coord in point]
#                 boxes.append(box)
#
#         if self.transform:
#             transformed = self.transform(image=np.array(image), bboxes=boxes, labels=[1] * len(boxes))
#             image = transformed["image"]
#             boxes = transformed["bboxes"]
#
#         # 归一化 boxes，方便计算损失
#         boxes = [[box[0] / width, box[1] / height, box[2] / width, box[3] / height] for box in boxes]
#
#         # 转换 boxes 到中心点-宽高格式
#         boxes = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]] for box in boxes]
#
#         label = [1] * len(boxes)
#
#         target = {
#             "labels": torch.tensor(label, dtype=torch.long),
#             "boxes": torch.tensor(boxes, dtype=torch.float32)
#         }
#
#         return image, target
#
#     def __len__(self):
#         return len(self.image_paths)
