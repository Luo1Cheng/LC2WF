import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class lossA(nn.Module):
    def __init__(self):
        super(lossA, self).__init__()
        pass
    def forward(self, output, target, indices):
        device = output['pred_point3D'].device
        idx = self._get_src_permutation_idx(indices)
        pred_point3D = output['pred_point3D'][idx] #pred_point3D: bs,100,3
        target_point3D = torch.cat([torch.Tensor(t['point_gt']).float().to(device)[i] for t,(_,i) in zip(target,indices)], dim=0)
        loss_point = F.mse_loss(pred_point3D,target_point3D, reduction='none')
        bs = output['pred_point3D'].size(0)
        num_bbox = sum([len(i['point_gt']) for i in target])
        num_bbox = torch.as_tensor([num_bbox], dtype=torch.float, device=device)

        src_logits = output['pred_logits']
        # target_class = torch.cat([t[i] for t,(_,i) in zip(batch['point_gt'],indices)], dim=0)
        target_class = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device = pred_point3D.device)
        target_class[idx] = 1.0
        loss_ce = F.cross_entropy(src_logits.transpose(1,2),target_class)

        loss_point = loss_point.sum() / num_bbox
        loss = loss_point + loss_ce * 10
        return {"loss_point":loss_point, "loss_ce": loss_ce, "loss": loss}

        # for i in range(output.size(0)):
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx




class MylossV2(nn.Module):
    '''
    基本和Myloss一样， 不过不用再算标签了， 直接使用之前算的标签。 注意所有loss输入的target 都是没有move到原点的
    '''
    def __init__(self):
        super(MylossV2, self).__init__()
        self.CE = torch.nn.CrossEntropyLoss()
        self.L1 = torch.nn.L1Loss(reduction='none')
    def forward(self, logits, predXYZ, targetXYZ, classifyLabel, fpsXYZ, stage="l2"):
        lossCls = self.CE(input=logits, target=classifyLabel)
        targetXYZ = targetXYZ - fpsXYZ
        targetXYZ = targetXYZ * classifyLabel.unsqueeze(-1)
        predXYZ = predXYZ.permute(0, 2, 1)
        predXYZ = predXYZ * classifyLabel.unsqueeze(-1)

        lossRuc = self.L1(predXYZ, targetXYZ)
        temp_out = lossRuc
        lossRuc = torch.sum(torch.sum(lossRuc, dim=-1) * classifyLabel, dim=-1)
        juncNum = torch.sum(classifyLabel, dim=-1)
        lossRuc = torch.mean(lossRuc / juncNum)
        temp_out = temp_out.sum(1) / juncNum.unsqueeze(-1)
        ret = {stage + "_lossCls": lossCls,
               stage + "_lossRuc": lossRuc,
               stage + "_loss": lossRuc * 10 + lossCls,
               stage + "_targetXYZ": targetXYZ,
               stage + "_predXYZ": predXYZ,
               stage + "_logits": logits,
               stage + "_mae_loss": temp_out,
                }
        return ret