import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from modeling.utils import *
from modeling.loss import *
# from modeling.point_transformer_pytorch import PointTransformerLayer



from modeling.transformer import TransformerEncoder, TransformerEncoderLayer
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


        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        # src = torch.max(src, dim=-1)[0]
        # src = torch.mean(src, dim=-1)

        bs, d, spb = src.shape
        src = src.permute(2,0,1)
        if pos_embed is not None: pos_embed = pos_embed.permute(2,0,1)


        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) # spb,bs,dim
        # ret hs  num_decoder * bs * dim
        return memory.permute(1, 2, 0)




class LineNetV2(torch.nn.Module):
    def __init__(self,cfg):
        super(LineNetV2,self).__init__()
        self.mlpList = nn.Sequential(
            nn.Conv3d(3, 64, 1, 1),  nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, 1, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 256, 1, 1), nn.BatchNorm3d(256), nn.ReLU(),
        )
        self.LA1 = LineNetAbstraction(128, 0.25, 16)
        self.Head1 = Head()
        self.Pos = PosEmb()
        self.Head2 = Head()
        self.Myloss = Myloss()
        self.MylossV2 = MylossV2()
        self.transformer = PointTransformerLayer(dim = 256, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4)
        # self.transformer = Transformer(d_model=256, nhead=4, num_encoder_layers=1,
        #          num_decoder_layers=1, dim_feedforward=1024, dropout=0.2,
        #          activation="relu", normalize_before=True,
        #          return_intermediate_dec=False )
    def forward(self,batch):
        l0_input = batch['input'] # 输入已经标准化， 并且减去了采样点，即平移到原点了
        l0_input = l0_input.permute(0,4,1,2,3)

        l1_points = self.mlpList(l0_input) # N,128,256,32,2
        l1_points = torch.max(l1_points, dim=-1)[0]
        l1_points = torch.mean(l1_points, dim=-1)

        #stage1
        l1_logits, l1_predXYZ = self.Head1(l1_points,l0_input,batch['fpsPoint']) #(N,2,256), (N,3,256)
        loss1 = self.Myloss(l1_logits, l1_predXYZ, batch['classifyLabel'], batch['label'], batch['objGTJunc3D'], batch['item'], batch['fpsPoint'],'l1')

        # src_targetXYZ = loss1['src_l1_targetXYZ'] * batch['classifyLabel'].unsqueeze(-1)

        AA = F.softmax(l1_logits,dim=1)
        pred = torch.where(AA[:,1,:]>0.5,1,0)
        src_targetXYZ = l1_predXYZ * pred.unsqueeze(1)
        src_targetXYZ = src_targetXYZ.permute(0,2,1)
        # posEmb = self.Pos(src_targetXYZ)

        # l1_predXYZ_2 = l1_predXYZ + batch['fpsPoint'].permute(0,2,1)

        #stage2
        # l2_points = self.transformer(l1_points, None, None, posEmb)
        l2_points = self.transformer(l1_points, src_targetXYZ, None)
        l2_logits, l2_predXYZ = self.Head2(l2_points, l0_input, batch['fpsPoint'])
        loss2 = self.MylossV2(l2_logits, l2_predXYZ, src_targetXYZ, batch['classifyLabel'], batch['fpsPoint'],'l2')

        loss1["l1_mae_loss"] = (loss1['l1_mae_loss'] * batch['std'].unsqueeze(-1)) #+ batch['mean']
        loss1["l1_mae_loss"] = loss1["l1_mae_loss"].mean()

        loss2["l2_mae_loss"] = (loss2['l2_mae_loss'] * batch['std'].unsqueeze(-1)) #+ batch['mean']
        loss2["l2_mae_loss"] = loss2["l2_mae_loss"].mean()


        loss={}
        loss['loss'] = loss1['l1_loss'] + 2*loss2['l2_loss']
        loss.update(loss1)
        loss.update(loss2)

        return loss


