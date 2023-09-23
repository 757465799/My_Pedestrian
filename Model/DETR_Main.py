# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops import MLP, box_area
import torch.nn.functional as F


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer):
        """ Initializes the model. """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        input_dim = 128  # 确保这是您模型的实际输出维度,输出到transformer
        MaxNum_query = 5
        # in_channels是backbone的输出维度
        self.tensor_Conv2d = nn.Conv2d(512, input_dim, kernel_size=(1, 1), stride=(1, 1))
        self.query_embed = nn.Embedding(MaxNum_query, input_dim)

        # 行人和背景类别
        num_classes = 1
        self.class_embed = nn.Linear(input_dim, num_classes + 1)

        # 4：边界框的四个坐标
        hidden_dims = [input_dim, 4]
        output_dim = 4
        self.bbox_embed = MLP(input_dim, hidden_dims + [output_dim])

        self.matcher = HungarianMatcher()

    def forward(self, images):
        padded_images, mask = pad_images_and_create_masks(images)
        # Stack the lists into tensors
        padded_images = torch.stack(padded_images, dim=0)
        mask = torch.stack(mask, dim=0)
        # backbone
        tensor, mask, pos = self.backbone(padded_images, mask)

        # 降维减少计算，得到更紧凑、更高效的特征
        tensor = self.tensor_Conv2d(tensor)

        # transformer
        hs = self.transformer(tensor, mask, self.query_embed.weight, pos)[0]  # query_embed：query,key,value

        # output
        # 分类
        outputs_class = self.class_embed(hs)  # 线性回归把特征输出到类别
        # 回归,中心点为模型提供了一个更为稳定和鲁棒的参考点.基于预测的偏移量来得到的,与Transformer的自注意机制相匹配
        outputs_coord = self.bbox_embed(hs).sigmoid()  # 用多层感知器映射到坐标空间 [x, y, width, height]，并把边框坐标转换成概率值 0~1
        # 大小匹配损失，去最后一层，它包含前面所有层的信息
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        return out


# padding, mask
def pad_images_and_create_masks(images, padding=16):
    padded_images = []
    masks = []

    for img in images:
        # Pad the image
        padded_img = F.pad(img, (padding, padding, padding, padding))
        padded_images.append(padded_img)

        # Create the mask
        mask = torch.zeros(padded_img.shape[1], padded_img.shape[2])
        mask[padding:-padding, padding:-padding] = 1  # 设置原始图像区域为1

        masks.append(mask)

    return padded_images, masks


# 损失函数
class Loss_dict(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.num_classes = 1
        self.loss_labels = loss_labels
        self.loss_boxes = loss_boxes
        self.model = model  # Store the model instance

    def forward(self, outputs, targets):
        outputs = {k: v for k, v in outputs.items()}

        # 1. Hungarian matching between output and label
        indices = self.model.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)

        # 2. Compute classification and bbox losses
        losses = {'loss_labels': self.loss_labels(outputs, targets, indices, self.num_classes)}

        # Note the modification here
        losses['loss_boxes'] = self.loss_boxes(outputs, targets, indices, num_boxes)

        return losses  # Return both losses and IoU scores


def loss_labels(outputs, targets, indices, num_classes):
    src_logits = outputs['pred_logits']

    idx = get_src_idx(indices)  # 获取匈牙利匹配结果后的重新排列的索引
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # 根据索引获取真实类别标签，一共有多少个匹配的目标
    # target_classes = torch.full(src_logits.shape[:2], num_classes, dtype=torch.int64,
    #                             device=src_logits.device)  # 创建背景类别,和预测输出大小相同，但少了一个维度
    target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)

    target_classes[idx] = target_classes_o  # 用真实的类别标签替换与目标匹配的预测的类别

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    empty_weight = torch.ones(num_classes + 1).to(device)
    loss_entropy = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)  # 计算交叉熵损失，背景预测不预测任何真实的对象
    losses = {'loss_entropy': loss_entropy}
    return losses


def loss_boxes(outputs, targets, indices, num_boxes):
    idx = get_src_idx(indices)
    src_boxes = outputs['pred_boxes'][idx]
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - torch.diag(generalized_box_iou(box_x1y1x2y2(src_boxes), box_x1y1x2y2(target_boxes)))
    losses['loss_giou'] = loss_giou.sum() / num_boxes

    return losses


# 匈牙利匹配
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network"""

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher"""
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching"""
        batch_size, MaxNum_query = outputs["pred_logits"].shape[:2]  # [batch_size, num_queries, num_classes]

        # Flatten the outputs to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # softmax 转换成概率得到类别
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # flatten bbox coordinates得到边界框

        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])

        # 三种损失：分类，边界框l1和giou
        # Classification cost approximated as 1 - proba[target class]

        # cost_class = out_prob[torch.arange(out_prob.shape[0]), tgt_ids]

        cost_class = -out_prob[:, tgt_ids]

        # 预测和真实边界框坐标点的L1距离
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # Generalized IoU (Intersection over Union) cost
        # print(box_x1y1x2y2(out_bbox))
        cost_giou = -generalized_box_iou(box_x1y1x2y2(out_bbox), box_x1y1x2y2(tgt_bbox))

        # 形成成本矩阵 Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, MaxNum_query, -1).cpu()

        # Number of targets per batch item
        num_targets_per_item = [len(v["boxes"]) for v in targets]
        # print('Number of targets per batch item', num_targets_per_item)
        # Compute optimal assignment using Hungarian algorithm
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(num_targets_per_item, -1))]

        # Return list of tuples containing matched prediction-target indices
        matched_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                           indices]
        return matched_indices


def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_x1y1x2y2(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def get_src_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
