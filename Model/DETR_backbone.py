import torch
import math

import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import models


class BackboneAndPos(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embedding = PositionEmbedding()
        self.backbone_model = MyBackbone()
        self.model = Joiner(self.backbone_model, self.position_embedding)

    def forward(self, images, mask):
        # 使用 self.model 获得所有阶段的输出
        tensor, mask, pos = self.model(images, mask)
        # 只保留最后一个阶段的输出
        # last_tensor = tensors[-1]
        # last_mask = masks[-1]
        # last_pos = pos[-1]
        return tensor, mask, pos


class Joiner(nn.Module):
    def __init__(self, backbone, position_embedding):
        super(Joiner, self).__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding

    def forward(self, padded_images, mask):
        # Forward pass through the backbone
        tensor, mask = self.backbone(padded_images, mask)

        # Position encoding
        pos = self.position_embedding((tensor, mask))

        return tensor, mask, pos


class MyBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # 使用预训练的 ResNet34 作为 backbone
        # self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = models.resnet50()

        # 替换 ResNet34 的全连接层和平均池化层为 Identity 层
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

    def forward(self, x, mask):
        x = self.backbone.conv1(x)
        mask = F.interpolate(mask[None].float(), size=x.shape[-2:], mode="nearest").to(torch.bool)[0]

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        mask = F.interpolate(mask[None].float(), size=x.shape[-2:], mode="nearest").to(torch.bool)[0]

        x = self.backbone.layer1(x)
        mask = F.interpolate(mask[None].float(), size=x.shape[-2:], mode="nearest").to(torch.bool)[0]

        x = self.backbone.layer2(x)
        mask = F.interpolate(mask[None].float(), size=x.shape[-2:], mode="nearest").to(torch.bool)[0]

        return x, mask


# 对不同位置进行编码，距离越远位置差异越大
class PositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64):  # num_pos_feats和backbone的输入维度匹配，输入维度的一半
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 位置特征的数量
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, images_mask):
        # assuming tensor_list is now a tuple: (tensors, mask)
        tensor, mask = images_mask
        # print(tensor.shape)
        # print(mask.shape)

        # 用mask计算编码可以捕获到图像的形状，边界等信息，比自己设置要好；他还考虑到了有效区域
        # assert mask is not None,由于1会累加，0不会，所以计算每个位置的累计和的时候，也就是连续坐标的时候我们取反
        not_mask = ~mask
        # 二维图像像素坐标为（y,x)，图像有效形状为(h,w)
        # 计算竖直方向上坐标
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # 计算水平方向上坐标
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # normalize 归一化，每个元素除以其所在行或列的最大值, 将其标准化:[0 , 2pi]
        # [批次:, 行:, 列:]
        eps = 1e-6
        ''' 
        
        y_embed = y/h * 2pi, x_embed = x/w * 2pi 
        
        (2pi 正弦余弦的周期，标准化在一定范围内避免梯度爆炸或消失)
        
        '''
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * (2 * math.pi)  # 除每个批次的最后一行
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * (2 * math.pi)  # 除每个批次中最后一列
        x_embed = x_embed.to(self.device)
        y_embed = y_embed.to(self.device)
        # print(x_embed.device)  # 打印 x_embed 的设备

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=self.device)  # 获取位置编码的维度,奇数还是偶数
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 调整dim_t的形状以匹配x_embed的广播规则
        # dim_t = dim_t[None, None, None, None, :]

        # 对x_embed和y_embed进行形状调整
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # 经过 sin 和 cos
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # 将 pos_y 和 pos_x 沿着通道维度（dim=1）合并，形状变为 [32, 512, 16, 16]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos
