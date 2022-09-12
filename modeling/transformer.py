# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from modeling.LineNet import Head,Myloss
import modeling.LineClassify as ML
class Mytransformer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        args = cfg['transformer']
        self.transformer = Transformer(
            d_model=args['hidden_dim'],
            dropout=args['dropout'],
            nhead=args['nheads'],
            dim_feedforward=args['dim_feedforward'],
            num_encoder_layers=args['num_encoder_layers_1'],
            num_decoder_layers=args['num_decoder_layers'],
            normalize_before=args['pre_norm'],
            return_intermediate_dec=True,)

        self.transformer2 = Transformer(
            d_model=args['hidden_dim'],
            dropout=args['dropout'],
            nhead=args['nheads'],
            dim_feedforward=args['dim_feedforward'],
            num_encoder_layers=args['num_encoder_layers_2'],
            num_decoder_layers=args['num_decoder_layers'],
            normalize_before=args['pre_norm'],
            return_intermediate_dec=True,)
        self.mlpList = nn.Sequential(
            nn.Conv2d(cfg['Net']['input_dim'], 64, 1, 1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.Head1 = Head([256, 128, 64, 32])
        self.Myloss = Myloss()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,batch, mode="train"):
        if mode=="train":
            return self.forward_train(batch)
        else:
            return self.forward_test(batch)
    def forward_train(self,batch):
        l0_input = batch['input']
        # l0_input = l0_input.flatten(start_dim=-2,end_dim=-1)
        bs,sn,gn,_ = l0_input.shape
        l0_input = l0_input.permute(0,3,1,2)
        l1_points = self.mlpList(l0_input)
        l1_points = l1_points.permute(0,2,3,1)
        bs,sn,gn,c = l1_points.shape
        l1_points = l1_points.view(bs*sn,gn,c)
        word_mask = batch['word_mask'].view(bs*sn,gn)
        l2_points,attention_weights_list = self.transformer(l1_points,word_mask,None,None)
        l2_points = l2_points.view(bs,sn,gn,-1)

        group_points = torch.mean(l2_points,dim=-2)
        # maybe cat some global infomation to group_points
        l3_points,attention_weights_list1 = self.transformer2(group_points,None,None,None)

        l3_points = l3_points.permute(0,2,1)
        l1_logits, l1_predXYZ = self.Head1(l3_points, None, batch['fpsPoint'])
        loss1 = self.Myloss(l1_logits, l1_predXYZ, batch['classifyLabel'], batch['label'], batch['objGTJunc3D'], batch['item'], batch['fpsPoint'],'l1')
        loss1["l1_mae_loss"] = (loss1['l1_mae_loss'] * batch['std'].unsqueeze(-1))
        loss1["l1_mae_loss"] = loss1["l1_mae_loss"].mean()

        loss = {}
        loss['loss'] = loss1['l1_loss']
        loss['attn1'] = torch.stack(attention_weights_list)
        loss['attn2'] = torch.stack(attention_weights_list1)
        loss.update(loss1)

        return loss

    def forward_test(self,batch):
        l0_input = batch['input']
        # l0_input = l0_input.flatten(start_dim=-2,end_dim=-1)
        bs,sn,gn,_ = l0_input.shape
        l0_input = l0_input.permute(0,3,1,2)
        l1_points = self.mlpList(l0_input)
        l1_points = l1_points.permute(0,2,3,1)
        bs,sn,gn,c = l1_points.shape
        l1_points = l1_points.view(bs*sn,gn,c)
        word_mask = batch['word_mask'].view(bs*sn,gn)
        l2_points, _ = self.transformer(l1_points,word_mask,None,None)
        l2_points = l2_points.view(bs,sn,gn,-1)

        group_points = torch.mean(l2_points,dim=-2)

        l3_points, _ = self.transformer2(group_points,None,None,None)

        l3_points = l3_points.permute(0,2,1)
        l1_logits, l1_predXYZ = self.Head1(l3_points, None, batch['fpsPoint'])

        loss = {
            "l1_predXYZ":l1_predXYZ,
            "l1_logits": l1_logits,
        }
        return loss

class Mytransformer_classify(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        args = cfg['transformer']
        self.transformer = Transformer(
            d_model=args['hidden_dim'],
            dropout=args['dropout'],
            nhead=args['nheads'],
            dim_feedforward=args['dim_feedforward'],
            num_encoder_layers=args['num_encoder_layers'],
            num_decoder_layers=args['num_decoder_layers'],
            normalize_before=args['pre_norm'],
            return_intermediate_dec=True,)

        self.transformer2 = Transformer(
            d_model=args['hidden_dim'],
            dropout=args['dropout'],
            nhead=args['nheads'],
            dim_feedforward=args['dim_feedforward'],
            num_encoder_layers=args['num_encoder_layers'],
            num_decoder_layers=args['num_decoder_layers'],
            normalize_before=args['pre_norm'],
            return_intermediate_dec=True,)

        self.mlpList = nn.Sequential(
            nn.Conv2d(cfg['Net']['input_dim'], 64, 1, 1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )

        # self.Head1 = Head([512, 256, 128, 64, 32])
        self.Head = ML.Head([256, 128, 64, 32])
        self.Myloss = ML.Myloss()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch, mode='train'):
        if mode=='train':
            return self.forward_train(batch)
        else:
            return self.forward_test(batch)
    def forward_train(self,batch):
        l0_input = batch['input']
        bs,sn,gn,_ = l0_input.shape
        l0_input = l0_input.permute(0,3,1,2)
        l1_points = self.mlpList(l0_input)
        l1_points = l1_points.permute(0,2,3,1)
        bs,sn,gn,c = l1_points.shape
        l1_points = l1_points.view(bs*sn,gn,c)
        word_mask = batch['word_mask'].view(bs*sn,gn)
        # word_mask = None
        l2_points,_ = self.transformer(l1_points,word_mask,None,None)
        l2_points = l2_points.view(bs,sn,gn,-1)

        group_points = torch.mean(l2_points,dim=-2)

        l3_points,_ = self.transformer2(group_points,None,None,None)

        l3_points = l3_points.permute(0,2,1)

        l1_logits = self.Head(l3_points)
        loss1 = self.Myloss(l1_logits, batch['classifyLabel'],'l1')

        loss={}
        loss['loss'] = loss1['l1_loss'] #+ edgeLoss
        #
        loss.update(loss1)

        return loss
        #

    def forward_test(self,batch):
        l0_input = batch['input']
        # l0_input = l0_input.flatten(start_dim=-2,end_dim=-1)
        bs,sn,gn,_ = l0_input.shape
        l0_input = l0_input.permute(0,3,1,2)
        l1_points = self.mlpList(l0_input)
        l1_points = l1_points.permute(0,2,3,1)
        bs,sn,gn,c = l1_points.shape
        l1_points = l1_points.view(bs*sn,gn,c)
        word_mask = batch['word_mask'].view(bs*sn,gn)
        # word_mask = None
        l2_points, _ = self.transformer(l1_points,word_mask,None,None)
        l2_points = l2_points.view(bs,sn,gn,-1)

        group_points = torch.mean(l2_points,dim=-2)
        l3_points, _ = self.transformer2(group_points,None,None,None)

        l3_points = l3_points.permute(0,2,1)

        l1_logits = self.Head(l3_points)

        loss={"l1_logits":l1_logits}

        return loss


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        # bs, sn, gn, c = src.shape
        # src = src.view(bs*sn,gn,c)
        bs, spb, d = src.shape
        src = src.permute(1,0,2)

        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # tgt = torch.zeros_like(query_embed)
        memory,attention_weights_list = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
        #                   pos=pos_embed, query_pos=query_embed)
        # ret hs  num_decoder * bs * dim
        memory = memory.permute(1, 0, 2)
        # memory = memory.view(bs,sn,gn,-1)
        return memory, attention_weights_list
        # return hs.transpose(1, 2), memory.permute(1, 0, 2)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_list = []
        for layer in self.layers:
            output,attn_weights = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            atten_list.append(attn_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output,atten_list


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2,attn_weight = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src,attn_weight

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args['hidden_dim'],
        dropout=args['dropout'],
        nhead=args['nheads'],
        dim_feedforward=args['dim_feedforward'],
        num_encoder_layers=args['num_encoder_layers'],
        num_decoder_layers=args['num_decoder_layers'],
        normalize_before=args['pre_norm'],
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
