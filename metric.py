import numpy as np
import glob
import cv2
import scipy.io as sio
import torch

def msTPFP(line_pred, line_gt, threshold):
    #输入 两个array  (N0,2,2) 和 (N1,2,2) ; (2,2)分别是 [(x1,y1),(x2,y2)]
    # print(line_pred[:,None,:,None].shape,line_gt[:,None].shape)
    diff = ((line_pred[:,None,:] - line_gt[None,:,:]) ** 2).sum(-1)


    #diff 为(N0,N1)  到这里计算了 pred的每个线 和gt的每个线 端点之间的距离的和(考虑已经算了左右顺序了)
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), np.bool_) # 记录每条线是否被用过
    tp = np.zeros(len(line_pred), np.float64)
    fp = np.zeros(len(line_pred), np.float64)
    # 每条预测线和真实线对 这里 线是sort过的 按照score， 因此 score高的会提前匹配
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]: # 这里是距离最小的，用条数，改成用对上最长的算像素
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1

    return tp, fp

class juncAP_woConfi:
    def __init__(self,s=7,nms_threshhold=5):
        self.s=s
        self.nms_threshhold = nms_threshhold
        self.tp_list=[]
        self.fp_list=[]
        self.score_list=[]
        self.len=0
        self.use=False
    def __call__(self,predXYZ,wireframeJunc):
        # 反归一化

        wireframeJunc = torch.Tensor(wireframeJunc).float()


        # cal recall
        predXYZ = torch.Tensor(predXYZ).float()
        gtXYZ = wireframeJunc
        # gtXYZ = wireframeJunc[0][rec_label[0]]

        N, _ = gtXYZ.shape  # N,3
        S, _ = predXYZ.shape  # S,3
        gtXYZ = gtXYZ.unsqueeze(1).repeat((1, S, 1))  # N,S,3
        positiveXYZ = predXYZ.unsqueeze(0).repeat((N, 1, 1))
        dist = torch.sum((gtXYZ - positiveXYZ) ** 2, dim=-1)  # N,S
        dist = torch.sqrt(dist).transpose(1, 0)  # S,N

        choice = torch.argmin(dist, dim=-1)
        dist = torch.min(dist, dim=-1)[0]
        hit = torch.zeros(N, dtype=torch.bool)
        tp = torch.zeros(dist.shape[0], dtype=torch.float64)
        fp = torch.zeros(dist.shape[0], dtype=torch.float64)

        for i in range(dist.shape[0]):
            if dist[i] < self.s and not hit[choice[i]]:
                hit[choice[i]] = True
                tp[i] = 1
            else:
                fp[i] = 1
        # 还没算ap 算个recall先
        self.tp_list.append(tp)
        self.fp_list.append(fp)
        self.score_list.append(tp)
        self.len+=N
    def __reset__(self):
        self.tp_list=[]
        self.fp_list=[]
        self.len=0
        self.score_list=[]
    def getap(self):
        if self.len==0:
            return 0
        tp_all = torch.cat(self.tp_list)
        fp_all = torch.cat(self.fp_list)
        score_all = torch.cat(self.score_list)
        score_index = torch.argsort(-score_all)
        tp_all = torch.cumsum(tp_all[score_index],dim=0) / self.len
        fp_all = torch.cumsum(fp_all[score_index],dim=0) / self.len

        return self.ap(tp_all,fp_all)
    def get_RPF(self):
        tp_all = torch.cat(self.tp_list)
        fp_all = torch.cat(self.fp_list)
        sum_tp = torch.sum(tp_all)
        sum_fp = torch.sum(fp_all)
        recall=sum_tp/self.len
        precision=sum_tp/(sum_fp+sum_tp)
        f1 = 2*recall*precision/(recall+precision)
        ret = {"recall":recall.item(),"precision":precision.item(),'f1':f1,'sum_tp':sum_tp,'sum_fp':sum_fp,'gt_num':self.len}
        return ret
        return recall,precision,f1,sum_tp,sum_fp,self.len
    def ap(self,tp, fp): #计算召回赫准确率, tp=G交P/G
        recall = tp
        precision = tp / np.maximum(tp + fp, 1e-9*torch.ones_like(tp)) # tp/P

        recall = torch.cat(( torch.Tensor([0.0]), recall, torch.Tensor([1.0]) )) #加上开头结尾
        precision = torch.cat((torch.Tensor([0.0]), precision, torch.Tensor([0.0])))

        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i]) # 是下降趋势
        i = torch.where(recall[1:] != recall[:-1])[0]
        return torch.sum((recall[i + 1] - recall[i]) * precision[i + 1]) #计算PR曲线下面积

    def print_result(self):
        d ={}
        d['Sap{}'.format(str(self.s))]=self.getap()
        adict = self.get_RPF()
        out = {**d,**adict}
        print(out)


class edgeSap_woConfi:
    def __init__(self,s=7,nms_threshhold=5):
        self.s=s
        self.nms_threshhold = nms_threshhold
        self.tp_list=[]
        self.fp_list=[]
        self.score_list=[]
        self.len=0
        self.use=False
    def __call__(self,predXYZ,combination,wireframeJunc,wireframeLine):
        def line_to_line_dist_torch(xx,yy):
            # 输入 两个array  (N0,2,2) 和 (N1,2,2) ; (2,2)分别是 [(x1,y1),(x2,y2)]
            diff = ((xx[:, None, :, None] - yy[:, None]) ** 2).sum(-1)
            # print(diff.shape)
            diff = torch.sqrt(diff)
            diff = torch.minimum(
                diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
            )
            return diff
        # 反归一化
        predXYZ = torch.Tensor(predXYZ).float()

        wireframeJunc = torch.Tensor(wireframeJunc)
        # fpsPoint = fpsPoint * max + mean
        # nms and confi-based mask

        combineXYZ = predXYZ[combination,:].tolist()


        # cal recall
        combineXYZ = torch.Tensor(combineXYZ).float()
        gtCombineXYZ = wireframeJunc[wireframeLine,:]
        # gtXYZ = wireframeJunc[0]
        # gtXYZ = wireframeJunc[0][rec_label[0]]

        N, _,_ = gtCombineXYZ.shape  # N,2,3
        S, _, _ = combineXYZ.shape  # S,2,3
        dist = line_to_line_dist_torch(gtCombineXYZ,combineXYZ).transpose(1,0)
        # gtCombineXYZ = gtCombineXYZ.unsqueeze(1).repeat((1, S, 1))  # N,S,3
        # positiveXYZ = combineXYZ.unsqueeze(0).repeat((N, 1, 1))
        # dist = torch.sum((gtCombineXYZ - positiveXYZ) ** 2, dim=-1)  # N,S
        # dist = torch.sqrt(dist).transpose(1, 0)  # S,N

        choice = torch.argmin(dist, dim=-1)
        dist = torch.min(dist, dim=-1)[0]
        hit = torch.zeros(N, dtype=torch.bool)
        tp = torch.zeros(dist.shape[0], dtype=torch.float64)
        fp = torch.zeros(dist.shape[0], dtype=torch.float64)

        for i in range(dist.shape[0]):
            if dist[i] < self.s and not hit[choice[i]]:
                hit[choice[i]] = True
                tp[i] = 1
            else:
                fp[i] = 1
        # 还没算ap 算个recall先
        self.tp_list.append(tp)
        self.fp_list.append(fp)
        self.score_list.append(tp)
        self.len+=N
        if tp.sum()/N<0.7:
            # print(tp.sum()/N)
            return True
    def __reset__(self):
        self.tp_list=[]
        self.fp_list=[]
        self.len=0
        self.score_list=[]
    def getap(self):
        if self.len==0:
            return 0
        tp_all = torch.cat(self.tp_list)
        fp_all = torch.cat(self.fp_list)
        score_all = torch.cat(self.score_list)
        score_index = torch.argsort(-score_all)
        tp_all = torch.cumsum(tp_all[score_index],dim=0) / self.len
        fp_all = torch.cumsum(fp_all[score_index],dim=0) / self.len

        return self.ap(tp_all,fp_all)
    def get_RPF(self):
        tp_all = torch.cat(self.tp_list)
        fp_all = torch.cat(self.fp_list)
        sum_tp = torch.sum(tp_all)
        sum_fp = torch.sum(fp_all)
        recall=sum_tp/self.len
        precision=sum_tp/(sum_fp+sum_tp)
        f1 = 2*recall*precision/(recall+precision)
        ret = {"recall":recall,"precision":precision,'f1':f1,'sum_tp':sum_tp,'sum_fp':sum_fp,'gt_num':self.len}
        return ret
        return recall,precision,f1,sum_tp,sum_fp,self.len
    def ap(self,tp, fp): #计算召回赫准确率, tp=G交P/G
        recall = tp
        precision = tp / np.maximum(tp + fp, 1e-9*torch.ones_like(tp)) # tp/P

        recall = torch.cat(( torch.Tensor([0.0]), recall, torch.Tensor([1.0]) )) #加上开头结尾
        precision = torch.cat((torch.Tensor([0.0]), precision, torch.Tensor([0.0])))

        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i]) # 是下降趋势
        i = torch.where(recall[1:] != recall[:-1])[0]
        return torch.sum((recall[i + 1] - recall[i]) * precision[i + 1]) #计算PR曲线下面积

    def print_result(self):
        d ={}
        d['Sap{}'.format(str(self.s))]=self.getap()
        adict = self.get_RPF()
        out = {**d,**adict}
        print(out)