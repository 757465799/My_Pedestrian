import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import torch

import matplotlib.pyplot as plt
from torch import device

from Model.DETR_Main import box_x1y1x2y2


def visualize_features(tensor, mask, pos):
    # 选择要可视化的特定通道
    channel_idx = 0

    # 获取特定通道的特征图
    feature_map = tensor[0, channel_idx].cpu().detach().numpy()

    # 获取掩码和pos
    mask_img = mask[0].cpu().detach().numpy()
    pos_img = pos[0, channel_idx].cpu().detach().numpy()

    # 绘图
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5))

    axarr[0].imshow(feature_map, cmap="viridis")
    axarr[0].set_title("Feature Map")

    axarr[1].imshow(mask_img, cmap="gray")
    axarr[1].set_title("Mask")

    axarr[2].imshow(pos_img, cmap="viridis")
    axarr[2].set_title("Pos")

    for ax in axarr:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# 反归一化
def denormalize(tensor, mean, std):
    tensor = tensor.permute(1, 2, 0)  # CxHxW -> HxWxC
    tensor = tensor * std + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor.numpy()


def visualize_padded_images_and_masks(images, masks):
    num_images = len(images)
    if num_images == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        axarr = np.array([[ax1, ax2]])
    else:
        fig, axarr = plt.subplots(num_images, 2, figsize=(10, num_images*5))

    for idx, (img, mask) in enumerate(zip(images, masks)):
        img = denormalize(img, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
        axarr[idx, 0].imshow(img)
        axarr[idx, 0].set_title("Padded Image")
        axarr[idx, 0].axis("off")

        axarr[idx, 1].imshow(mask, cmap="gray")
        axarr[idx, 1].set_title("Mask")
        axarr[idx, 1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_predictions(image, predicted_boxes, predicted_classes, image_width, image_height, class_names=None):
    """
    Plot all the predicted boxes on the image.

    :param image: torch.Tensor of shape [3, H, W] or numpy array of shape [H, W, 3]
    :param predicted_boxes: numpy array of shape [N, 4], where N is the number of boxes.
    :param predicted_classes: numpy array of shape [N], where N is the number of boxes.
    :param image_width: Original width of the image
    :param image_height: Original height of the image
    :param class_names: (Optional) List of class names to show instead of class IDs.
    """

    # Convert the image from torch.Tensor to numpy array if necessary
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    # Check if normalization is required
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(image)

    for box, label in zip(predicted_boxes, predicted_classes):
        x, y, w, h = box
        x, y = x - w / 2, y - h / 2  # 将框从中心点-宽高格式转回到左上角-宽高格式
        rect = plt.Rectangle((x * image_width, y * image_height), w * image_width, h * image_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x * 128, y * 128, str(label), fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()
    ax.set_title("Predicted Boxes")
    plt.tight_layout()
    plt.show()


def visualize_predictions_from_loader(model, loader, plot_function, img_width, img_height):
    """
    从数据加载器中随机选择一个样本并可视化模型的预测。

    参数:
        model: 模型对象
        loader: 数据加载器
        plot_function: 用于绘制预测的函数
        img_width: 图像的宽度
        img_height: 图像的高度
    """

    # 从数据加载器中随机选择一个批次的数据
    images, targets = next(iter(loader))

    # 随机选择一个图像
    random_index = torch.randint(len(images), size=(1,)).item()
    sample_image = images[random_index].to(device).unsqueeze(0)
    sample_target = targets[random_index]
    print("True class labels for the sample image:", sample_target)

    # 通过模型执行推理
    with torch.no_grad():
        outputs = model(sample_image)

    # 将输出转换为numpy数组
    predicted_box = outputs['pred_boxes'][0].cpu().numpy()
    predicted_classes = outputs['pred_logits'].softmax(-1)[0].argmax(-1).cpu().numpy()

    # 使用提供的函数绘制预测
    plot_function(sample_image.squeeze(0), predicted_box, predicted_classes, img_width, img_height)

#
# # colors for visualization
# COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]
#
# # COCO classes
# CLASSES = [
#     'background', 'pedestrian'
# ]
#
#
# def plot_results(pil_img, prob, boxes):
#     plt.figure(figsize=(16,10))
#     plt.imshow(pil_img)
#     ax = plt.gca()
#     colors = COLORS * 100
#     for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=c, linewidth=3))
#         cl = p.argmax()
#         text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
#         ax.text(xmin, ymin, text, fontsize=15,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
#     plt.axis('off')
#     plt.show()
#
# def rescale_bboxes(out_bbox, size):
#     img_w, img_h = size
#     b = box_x1y1x2y2(out_bbox)
#     b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
#     return b
