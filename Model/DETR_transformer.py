from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
import torch.nn.functional as F


class Transformer(nn.Module):

    def __init__(self, d_model=128, nhead=2, num_encoder_layers=4
                 ,
                 num_decoder_layers=4, dim_feedforward=128, dropout=0.2,
                 activation="relu", return_intermediate_dec=False):  # return_intermediate_dec=False 只返回最后一层输出
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                          return_intermediate=return_intermediate_dec)

    def forward(self, tensor, mask, query_embed, pos_embed):
        bs, c, h, w = tensor.shape

        # [batch_size, channels, height, width] ---> [height*width, batch_size, channels]
        tensor = tensor.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        device = torch.device("cuda:0")
        tensor = tensor.to(device)
        pos_embed = pos_embed.to(device)
        query_embed = query_embed.to(device)
        mask = mask.to(device)

        # print(tensor.shape, mask.shape, query_embed.shape, pos_embed.shape)

        target = torch.zeros_like(query_embed)  # 创建形状与 query_embed 相同的全0的向量作为作为encoder的输入，预测类别和位置

        # encoder
        memory = self.encoder(tensor, tensor_key_padding_mask=mask, pos=pos_embed)  # 编码器处理特征得到全局特征表示

        # decoder，和传统的transformer不同，它一次性处理全部query_embed，并输出全部的预测
        hs = self.decoder(target, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        # print(hs.transpose(1, 2).shape)
        # print(memory.permute(1, 2, 0).view(bs, c, h, w).shape)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, tensor, mask: Optional[Tensor] = None, tensor_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = tensor
        for layer in self.layers:
            output = layer(output, tensor_mask=mask, tensor_key_padding_mask=tensor_key_padding_mask, pos=pos)

        return output


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tensor,
                tensor_mask: Optional[Tensor] = None,  # 用在nlp中遮盖下一个单词，但在detr中因为直接预测全部数据，所以不需要
                tensor_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        tensor2 = self.norm1(tensor)  # 对tensor进行层归一化
        q = k = with_pos_embed(tensor2, pos)  # query 和 key添加位置编码, value不需要：tensor2和位置编码相加
        # query (q) 和 key (k) 之间的点积决定了注意力的权重分配，这些权重将会与 value (v) 相乘来得到输出
        tensor2 = self.self_attn(q, k, value=tensor2, attn_mask=tensor_mask, key_padding_mask=tensor_key_padding_mask)[
            0]  # padding mask将与填充位置对应的注意力权重设置为一个非常大的负数来实现的，然后应用softmax。在softmax函数中，一个非常大的负数近似为0
        tensor = tensor + self.dropout1(tensor2)  # tensor和计算好的tensor2相加，形成残差连接。dropout正则化防止过拟合
        tensor2 = self.norm2(tensor)
        tensor2 = self.linear2(self.dropout(self.activation(self.linear1(tensor2))))  # FFN
        tensor = tensor + self.dropout2(tensor2)
        tensor = self.norm3(tensor)
        return tensor


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.return_intermediate = return_intermediate

    def forward(self, target, memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                target_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = target

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, target_mask=target_mask,
                           memory_mask=memory_mask,
                           target_key_padding_mask=target_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, target, memory,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                target_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = k = with_pos_embed(target, query_pos)  # target,也就是和query_embed形状一样的全是0的向量作为输入，和10个查询对象的相对位置特征相加

        # self-attention
        target2 = self.self_attn(q, k, value=target, attn_mask=target_mask, key_padding_mask=target_key_padding_mask)[0]
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        query = with_pos_embed(target, query_pos)
        key = with_pos_embed(memory, pos)
        # multi-head attention
        target2 = self.multihead_attn(query=query, key=key, value=memory, attn_mask=memory_mask,
                                      key_padding_mask=memory_key_padding_mask)[0]
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # feedforward
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target))))
        target = target + self.dropout3(target2)
        target = self.norm3(target)

        return target
