import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from modeling.model_utils import classify, Mlp
'''
为实验BCEloss

'''


class Head(nn.Module):
    def __init__(self, mlp=[256, 128,64,32]):
        super(Head, self).__init__()
        self.classify1 = classify(mlp=mlp+[1])
        # self.rec1 = recNet()
        self.rec1 = classify(mlp=mlp+[3])
    def forward(self, points, xyz, fpsPoint): # points: N,C,sp
        # l1_predXYZ = self.rec1(points, xyz)

        # fpsPoint = fpsPoint.permute(0,2,1)
        # points = torch.cat([points,fpsPoint], dim=1)

        l1_logits = self.classify1(points)
        l1_predXYZ = self.rec1(points)
        return l1_logits,l1_predXYZ

class HeadEdge(nn.Module):
    def __init__(self):
        super(HeadEdge, self).__init__()
        self.classify1 = Mlp(mlp=[256, 128,64,32,1])
        self.BCE_loss =nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, points, heatMap, mask, positiveMask): # points: N,C,sp
        points = points.permute(0,2,1) # N,sp,C
        N,sp,C = points.shape
        X = points[:,:,None,:] + points[:,None,:,:] # N,sp,sp,C
        # X1 = points.unsqueeze(2).repeat([1, 1, sp, 1])
        # X2 = points.unsqueeze(1).repeat([1, sp, 1, 1])
        # X = torch.cat([X1,X2],dim=-1)

        predHeatMap = self.classify1(X).squeeze(-1)
        loss = self.BCE_loss(predHeatMap,heatMap)
        loss = loss * mask * positiveMask
        loss = loss.sum()/positiveMask.sum()
        return predHeatMap,loss

class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.L1 = torch.nn.L1Loss(reduction='none')
    def forward(self, logits, predXYZ, classifyLabel, label, objGTJunc3D, item, fpsXYZ, stage="l1"):
        lossCls = self.BCE(input=logits, target=classifyLabel.unsqueeze(1))
        classifyMask = torch.where(label >= 0, torch.ones_like(label), torch.zeros_like(label)).to(classifyLabel.device)
        lossCls = lossCls * ( classifyMask*2 + 1).unsqueeze(1)
        lossCls = lossCls.mean()
        device = label.device
        l1_targetXYZ = []
        for i in range(predXYZ.shape[0]):
            l1_targetXYZ.append(torch.Tensor(objGTJunc3D[item[i]]).float().to(device)[label[i]])
        l1_targetXYZ = torch.stack(l1_targetXYZ, dim=0)
        src_l1_targetXYZ = l1_targetXYZ
        l1_targetXYZ = l1_targetXYZ - fpsXYZ

        predXYZ = predXYZ.permute(0, 2, 1)


        lossRuc = self.L1(predXYZ, l1_targetXYZ) * classifyMask.unsqueeze(-1)
        temp_out = lossRuc
        lossRuc = torch.sum(torch.sum(lossRuc, dim=-1) * classifyMask, dim=-1)
        juncNum = torch.sum(classifyMask, dim=-1)
        lossRuc = torch.mean(lossRuc / juncNum)
        temp_out = temp_out.sum(1) / juncNum.unsqueeze(-1)
        ret =  {stage+"_lossCls":lossCls,
                stage+"_lossRuc":lossRuc,
                stage+"_loss": lossRuc*3 + lossCls,
                stage+"_targetXYZ":l1_targetXYZ,
                stage+"_predXYZ": predXYZ, #这里我写错拉！原本我写的是predXYZ，但应该是src_predXYZ
                stage+"_logits": logits,
                stage+"_mae_loss":temp_out,
                "src_l1_targetXYZ":src_l1_targetXYZ}
        return ret


class LineNet(torch.nn.Module):
    def __init__(self,cfg):
        super(LineNet,self).__init__()
        self.mlpList = nn.Sequential(
            nn.Conv3d(3, 64, 1, 1),  nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, 1, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 256, 1, 1), nn.BatchNorm3d(256), nn.ReLU(),
        )

        self.Head1 = Head()
        self.Myloss = Myloss()

    def forward(self,batch):
        l0_input = batch['input']
        l0_input = l0_input.permute(0,4,1,2,3)
        device = l0_input.device

        l1_points = self.mlpList(l0_input) # N,128,256,32,2
        l1_points = torch.max(l1_points, dim=-1)[0]
        l1_points = torch.mean(l1_points, dim=-1)


        #stage1
        l1_logits, l1_predXYZ = self.Head1(l1_points,l0_input,batch['fpsPoint'])
        loss1 = self.Myloss(l1_logits, l1_predXYZ, batch['classifyLabel'], batch['label'], batch['objGTJunc3D'], batch['item'], batch['fpsPoint'],'l1')


        loss1["l1_mae_loss"] = (loss1['l1_mae_loss'] * batch['std'].unsqueeze(-1)) #+ batch['mean']
        loss1["l1_mae_loss"] = loss1["l1_mae_loss"].mean()


        loss={}
        loss['loss'] = loss1['l1_loss'] #+ edgeLoss
        loss.update(loss1)

        return loss


class LineNetGlobal(torch.nn.Module):
    def __init__(self,cfg):
        super(LineNetGlobal,self).__init__()
        self.mlpList = nn.Sequential(
            nn.Conv3d(3, 64, 1, 1),  nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, 1, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 256, 1, 1), nn.BatchNorm3d(256), nn.ReLU(),
        )

        self.Head1 = Head([515, 256, 128,64,32])
        self.Myloss = Myloss()
        self.mlpList2 = nn.Sequential(
            nn.Conv3d(3, 64, 1, 1),  nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, 1, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 128, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 256, 1, 1), nn.BatchNorm3d(256), nn.ReLU(),
        )

    def forward(self,batch):
        l0_input = batch['input']
        l0_input = l0_input.permute(0,4,1,2,3)
        device = l0_input.device

        l1_points = self.mlpList(l0_input) # N,128,256,32,2
        l1_points = torch.max(l1_points, dim=-1)[0]
        l1_points = torch.mean(l1_points, dim=-1)

        global_feature = self.mlpList2(l0_input)
        global_feature = torch.max(global_feature, dim=-1)[0]
        global_feature = torch.mean(global_feature, dim=-1)
        global_feature = torch.max(global_feature, dim=-1)[0]
        global_feature = global_feature.unsqueeze(-1)
        global_feature = global_feature.repeat((1,1,l1_points.shape[-1]))
        # l1_points = torch.cat([l1_points, global_feature], dim=1)


        tempFps = batch['fpsPoint'].permute(0,2,1)
        l1_points = torch.cat([tempFps,l1_points,global_feature],dim=1)
        #stage1
        l1_logits, l1_predXYZ = self.Head1(l1_points,l0_input,batch['fpsPoint'])
        loss1 = self.Myloss(l1_logits, l1_predXYZ, batch['classifyLabel'], batch['label'], batch['objGTJunc3D'], batch['item'], batch['fpsPoint'],'l1')



        loss1["l1_mae_loss"] = (loss1['l1_mae_loss'] * batch['std'].unsqueeze(-1)) #+ batch['mean']
        loss1["l1_mae_loss"] = loss1["l1_mae_loss"].mean()

        loss={}
        loss['loss'] = loss1['l1_loss'] #+ edgeLoss

        loss.update(loss1)

        return loss
