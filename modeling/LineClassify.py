import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from modeling.model_utils import classify, Mlp

class Head(nn.Module):
    def __init__(self, mlp=[256, 128,64,32]):
        super(Head, self).__init__()
        self.classify1 = classify(mlp=mlp+[5])
        # self.rec1 = recNet()
        # self.rec1 = classify(mlp=mlp+[3])
    def forward(self, points): # points: N,C,sp
        # l1_predXYZ = self.rec1(points, xyz)

        # fpsPoint = fpsPoint.permute(0,2,1)
        # points = torch.cat([points,fpsPoint], dim=1)

        l1_logits = self.classify1(points)
        # l1_predXYZ = self.rec1(points)
        return l1_logits

class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.weight = torch.Tensor([1,3,1,1,1])
        # self.weight = torch.Tensor([1, 5])
        self.CE = torch.nn.CrossEntropyLoss(weight=self.weight)
    def forward(self, logits, classifyLabel, stage="l1"):

        lossCls = self.CE(input=logits, target=classifyLabel.long())
        device = logits.device

        ret =  {stage+"_lossCls":lossCls,
                stage + "_loss": lossCls,
                stage+"_logits": logits,}
        return ret


class ClassifyNetGlobal(torch.nn.Module):
    def __init__(self,cfg):
        super(ClassifyNetGlobal,self).__init__()
        self.mlpList = nn.Sequential(
            nn.Conv3d(3, 64, 1, 1),  nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, 1, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 256, 1, 1), nn.BatchNorm3d(256), nn.ReLU(),
        )

        self.Head1 = Head([512, 256, 128,64,32])
        self.Myloss = Myloss()
        self.mlpList2 = nn.Sequential(
            nn.Conv3d(3, 64, 1, 1),  nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, 1, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 256, 1, 1), nn.BatchNorm3d(256), nn.ReLU(),
        )

    def forward(self,batch):
        l0_input = batch['input']# bs,128,32,2,3
        l0_input = l0_input.permute(0,4,1,2,3) # bs,3,128,32,2
        device = l0_input.device

        l1_points = self.mlpList(l0_input) # N,256,128,32,2
        l1_points = torch.max(l1_points, dim=-1)[0]
        l1_points = torch.mean(l1_points, dim=-1) # bs,256,128

        global_feature = self.mlpList2(l0_input)
        global_feature = torch.max(global_feature, dim=-1)[0]
        global_feature = torch.mean(global_feature, dim=-1)
        global_feature = torch.max(global_feature, dim=-1)[0]
        global_feature = global_feature.unsqueeze(-1)
        global_feature = global_feature.repeat((1,1,l1_points.shape[-1]))
        l1_points = torch.cat([l1_points,global_feature],dim=1)
        #stage1
        l1_logits = self.Head1(l1_points)
        loss1 = self.Myloss(l1_logits, batch['classifyLabel'],'l1')
        #
        #
        #
        # loss1["l1_mae_loss"] = (loss1['l1_mae_loss'] * batch['std'].unsqueeze(-1)) #+ batch['mean']
        # loss1["l1_mae_loss"] = loss1["l1_mae_loss"].mean()
        #
        loss={}
        loss['loss'] = loss1['l1_loss'] #+ edgeLoss
        #
        loss.update(loss1)

        return loss
