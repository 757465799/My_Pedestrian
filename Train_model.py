import os

import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from torch import device

from Dataset_Process import load_dataset
from Model.DETR_Main import DETR, Loss_dict, pad_images_and_create_masks
from Model.DETR_backbone import BackboneAndPos
from Model.DETR_transformer import Transformer
from tqdm import tqdm
# from mAP import calculate_image_precision, calculate_precision
from mean_average_precision import mean_average_precision_2d
from torch.utils.tensorboard import SummaryWriter
from mAP import calculate_precision
from visual import visualize_padded_images_and_masks, visualize_features, plot_predictions


def train_one_epoch(model, train_loader, optimizer, loss_dict, device,
                    total_mAP, total_loss, bbox_loss, giou_loss, labels_loss):
    model.train()
    train_pbar = tqdm(train_loader, desc="Training", position=0, leave=True)

    # Reset all AverageMeters
    total_mAP.reset()
    total_loss.reset()
    bbox_loss.reset()
    giou_loss.reset()
    labels_loss.reset()

    # # 使用第一个batch的数据进行可视化padding
    # first_batch_images, first_batch_targets = next(iter(train_loader))
    # first_image = [first_batch_images[0]]  # 选择第一张图像
    # padded_images_s, masks_s = pad_images_and_create_masks(first_image)  # 这里只获取masks，因为在模型中已经进行了padding
    # visualize_padded_images_and_masks(padded_images_s, masks_s)
    #
    # # 可视化经过backbone的特征图
    # tensor, mask, pos = model.backbone(padded_images_s[0].unsqueeze(0).to(device), masks_s[0].unsqueeze(0).to(device))
    # # 调用函数可视化backbone
    # visualize_features(tensor, mask, pos)

    for images, targets in train_pbar:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        outputs = model(images)

        loss_dict_values = loss_dict(outputs, targets)

        # Combine the losses:
        losses = sum(loss for sub_dict in loss_dict_values.values() for loss in sub_dict.values())

        # 计算 mAP（使用 final_pred_boxes 和 final_pred_scores）
        mAP_batch = []
        for i in range(len(targets)):
            gts = targets[i]["boxes"].detach().cpu().numpy()  # COCO: (cx, cy, w, h)
            # Convert to Pascal VOC: (xmin, ymin, xmax, ymax)
            gts[:, 0] -= gts[:, 2] / 2
            gts[:, 1] -= gts[:, 3] / 2
            gts[:, 2] += gts[:, 0]
            gts[:, 3] += gts[:, 1]

            preds = outputs["pred_boxes"][i].detach().cpu().numpy()  # COCO: (cx, cy, w, h)
            # Convert to Pascal VOC: (xmin, ymin, xmax, ymax)
            preds[:, 0] -= preds[:, 2] / 2
            preds[:, 1] -= preds[:, 3] / 2
            preds[:, 2] += preds[:, 0]
            preds[:, 3] += preds[:, 1]

            img_AP = calculate_precision(gts, preds, threshold=0.5, form="pascal_voc", ious=None)
            mAP_batch.append(img_AP)

        mAP = np.array(mAP_batch).mean()

        # BP
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # update loss & mAP per BATCH
        total_loss.update(losses.item())
        bbox_loss.update(loss_dict_values['loss_boxes']['loss_bbox'].item())
        giou_loss.update(loss_dict_values['loss_boxes']['loss_giou'].item())
        labels_loss.update(loss_dict_values['loss_labels']['loss_entropy'].item())
        total_mAP.update(mAP)

        # update progress bar
        train_pbar.set_postfix(bbox_loss=bbox_loss.avg, giou_loss=giou_loss.avg, labels_loss=labels_loss.avg,
                               total_loss=total_loss.avg, mAP=total_mAP.avg)
        pass
    pass
    return total_loss, total_mAP


def validate_one_epoch(model, val_loader, loss_dict, device, total_mAP, total_loss, bbox_loss, giou_loss, labels_loss):
    model.eval()
    total_mAP.reset()
    total_loss.reset()
    bbox_loss.reset()
    giou_loss.reset()
    labels_loss.reset()
    val_pbar = tqdm(val_loader, desc="Validation", position=0, leave=True)

    with torch.no_grad():
        for images, targets in val_pbar:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            outputs = model(images)

            loss_dict_values = loss_dict(outputs, targets)

            losses = sum(loss for sub_dict in loss_dict_values.values() for loss in sub_dict.values())

            # 计算 mAP（使用 final_pred_boxes 和 final_pred_scores）
            mAP_batch = []
            for i in range(len(targets)):
                gts = targets[i]["boxes"].detach().cpu().numpy()  # COCO: (cx, cy, w, h)
                # Convert to Pascal VOC: (xmin, ymin, xmax, ymax)
                gts[:, 0] -= gts[:, 2] / 2
                gts[:, 1] -= gts[:, 3] / 2
                gts[:, 2] += gts[:, 0]
                gts[:, 3] += gts[:, 1]

                preds = outputs["pred_boxes"][i].detach().cpu().numpy()  # COCO: (cx, cy, w, h)
                # Convert to Pascal VOC: (xmin, ymin, xmax, ymax)
                preds[:, 0] -= preds[:, 2] / 2
                preds[:, 1] -= preds[:, 3] / 2
                preds[:, 2] += preds[:, 0]
                preds[:, 3] += preds[:, 1]

                img_AP = calculate_precision(gts, preds, threshold=0.5, form="pascal_voc", ious=None)
                mAP_batch.append(img_AP)

            mAP = np.array(mAP_batch).mean()

            # update loss & mAP per BATCH
            total_loss.update(losses.item())
            bbox_loss.update(loss_dict_values['loss_boxes']['loss_bbox'].item())
            giou_loss.update(loss_dict_values['loss_boxes']['loss_giou'].item())
            labels_loss.update(loss_dict_values['loss_labels']['loss_entropy'].item())
            total_mAP.update(mAP)
            val_pbar.set_postfix(total_loss=total_loss.avg, mAP=total_mAP.avg)
            pass
        pass
        return total_loss, total_mAP


def test_model(model, test_loader, loss_dict, device, total_mAP, total_loss, bbox_loss, giou_loss, labels_loss):
    model.eval()
    val_loss = 0
    total_mAP.reset()
    total_loss.reset()
    bbox_loss.reset()
    giou_loss.reset()
    labels_loss.reset()
    test_pbar = tqdm(test_loader, desc="Validation", position=0, leave=True)

    with torch.no_grad():
        for images, targets in test_pbar:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            outputs = model(images)

            loss_dict_values = loss_dict(outputs, targets)

            losses = sum(loss for sub_dict in loss_dict_values.values() for loss in sub_dict.values())

            # 计算 mAP（使用 final_pred_boxes 和 final_pred_scores）
            mAP_batch = []
            for i in range(len(targets)):
                gts = targets[i]["boxes"].detach().cpu().numpy()  # COCO: (cx, cy, w, h)
                # Convert to Pascal VOC: (xmin, ymin, xmax, ymax)
                gts[:, 0] -= gts[:, 2] / 2
                gts[:, 1] -= gts[:, 3] / 2
                gts[:, 2] += gts[:, 0]
                gts[:, 3] += gts[:, 1]

                preds = outputs["pred_boxes"][i].detach().cpu().numpy()  # COCO: (cx, cy, w, h)
                # Convert to Pascal VOC: (xmin, ymin, xmax, ymax)
                preds[:, 0] -= preds[:, 2] / 2
                preds[:, 1] -= preds[:, 3] / 2
                preds[:, 2] += preds[:, 0]
                preds[:, 3] += preds[:, 1]

                img_AP = calculate_precision(gts, preds, threshold=0.5, form="pascal_voc", ious=None)
                mAP_batch.append(img_AP)

            mAP = np.array(mAP_batch).mean()

            # update loss & mAP per BATCH
            total_loss.update(losses.item())
            bbox_loss.update(loss_dict_values['loss_boxes']['loss_bbox'].item())
            giou_loss.update(loss_dict_values['loss_boxes']['loss_giou'].item())
            labels_loss.update(loss_dict_values['loss_labels']['loss_entropy'].item())
            total_mAP.update(mAP)
            test_pbar.set_postfix(total_loss=total_loss.avg, mAP=total_mAP.avg)
    return total_loss, total_mAP


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "C:/Users/h/pythonProject/LabelmeToCoco/annotations/"

    train_loader, val_loader, test_loader = load_dataset(dataset_path)

    backbone = BackboneAndPos()
    transformer = Transformer()
    DETR_Model = DETR(backbone, transformer).to(device)

    loss_dict = Loss_dict(DETR_Model).to(device)
    optimizer = torch.optim.AdamW(DETR_Model.parameters(), lr=0.001, betas=(0.9, 0.999))
    total_mAP = AverageMeter()
    total_loss = AverageMeter()
    bbox_loss = AverageMeter()
    giou_loss = AverageMeter()
    labels_loss = AverageMeter()
    # create folder to store model weights
    output_folder = "./output/"
    os.makedirs(output_folder, exist_ok=True)

    # Initiate TensorBoard
    tb_writer = SummaryWriter(comment="my_experiment")
    best_loss = 10 ** 5
    best_mAP = -1.0
    epochs = 100
    for epoch in range(epochs):
        train_loss, train_mAP = train_one_epoch(DETR_Model, train_loader, optimizer, loss_dict, device, total_mAP, total_loss, bbox_loss, giou_loss, labels_loss)

        val_loss, val_mAP = validate_one_epoch(DETR_Model, val_loader, loss_dict, device, total_mAP, total_loss, bbox_loss, giou_loss, labels_loss)
        print('|EPOCH %d| TRAIN_LOSS %.3f| VALID_LOSS %.3f| TRAIN_mAP@0.5 %.4f| VAL_mAP@0.5 %.4f|\n' %
              (epoch + 1, train_loss.avg, val_loss.avg, train_mAP.avg, val_mAP.avg))
        if tb_writer:
            tags = ['loss/train_loss', "loss/val_loss", "mAP/train_mAP@0.5", "mAP/val_mAP@0.5"]
            for x, tag in zip([train_loss.avg, val_loss.avg, train_mAP.avg, val_mAP.avg], tags):
                tb_writer.add_scalar(tag, x, epoch)

        # 检测验证集的预测
        sample_images, sample_targets = next(iter(val_loader))
        random_idx = torch.randint(len(sample_images), size=(1,)).item()
        sample_image = sample_images[random_idx].to(device).unsqueeze(0)
        sample_image_class_labels = sample_targets[random_idx]['labels'].cpu().numpy()

        # 打印出类别
        print("True class labels for the sample image:", sample_image_class_labels)

        with torch.no_grad():
            sample_outputs = DETR_Model(sample_image)
        predicted_boxes = sample_outputs['pred_boxes'][0].cpu().numpy()
        predicted_classes = sample_outputs['pred_logits'].softmax(-1)[0].argmax(-1).cpu().numpy()
        print("True class labels for the predict image:", predicted_classes)

        plot_predictions(sample_image.squeeze(0), predicted_boxes, predicted_classes, 1280, 720)

        # Visualization of predictions on a random validation image after each epoch
        if val_mAP.avg > best_mAP:
            best_loss = val_loss.avg
            best_mAP = val_mAP.avg
            print(
                f"Best model so far, VALID_LOSS {best_loss:.3f}, VAL_mAP@0.5 {best_mAP:.4f} in Epoch {epoch + 1}|...Saving Model...\n")
            torch.save(DETR_Model.state_dict(), f'{output_folder}/best_at_Epoch{epoch}.pth')

        # save the last epoch model
        if epoch + 1 == epochs:
            print(
                f"Last model, VALID_LOSS {best_loss:.3f}, VAL_mAP@0.5 {best_mAP:.4f} in Epoch {epoch + 1}|...Saving Model...\n")
            torch.save(DETR_Model.state_dict(), f'{output_folder}/last.pth')

    test_loss, test_mAP = test_model(DETR_Model, test_loader, loss_dict, device, total_mAP, total_loss, bbox_loss, giou_loss, labels_loss)
    print(f"Final Test Loss: {test_loss.avg}")
    print(f"Final Test mAP: {test_mAP.avg}")

    # (Your existing code for plotting, etc.)
    # 从测试数据加载器中随机选择一个批次的数据
    images, targets = next(iter(test_loader))

    # 随机选择一个图像
    random_index = torch.randint(len(images), size=(1,)).item()
    sample_image = images[random_index].to(device).unsqueeze(0)  # 添加批次维度
    sample_target = targets[random_index]
    print("True class labels for the sample image:", sample_target)

    # 通过模型执行推理
    with torch.no_grad():
        outputs = DETR_Model(sample_image)

    # 将输出转换为numpy数组
    predicted_box = outputs['pred_boxes'][0].cpu().numpy()
    predicted_classes = outputs['pred_logits'].softmax(-1)[0].argmax(-1).cpu().numpy()

    # Use the above function to plot the predictions
    plot_predictions(sample_image.squeeze(0), predicted_box, predicted_classes, 1280, 720)


if __name__ == '__main__':
    main()

