import torch
import numpy as np
import os
import torch.nn.functional as F
from progress.bar import Bar
from utils.utils import AverageMeter
import json

def nms(pred,confi,*args,**kwargs):
    all = list(zip(pred,confi,*args))
    pred,confi,*args = zip(*sorted(all, reverse=True, key=lambda x:x[1]))

    predArray = np.array(pred,dtype=np.float64)
    dropped_junc_index = []
    nms_threshhold = 5 if "nms_threshhold" not in kwargs.keys() else kwargs['nms_threshhold']
    for j in range(len(all)):
        if j in dropped_junc_index:
            continue
        dist_all = np.linalg.norm(predArray - predArray[j], axis=1)
        same_region_indexes = (dist_all < nms_threshhold).nonzero()
        for same_region_index_i in same_region_indexes[0]:
            if same_region_index_i == j:
                continue
            if confi[same_region_index_i] <= confi[j]:
                dropped_junc_index.append(same_region_index_i)
            else:
                dropped_junc_index.append(j)
    all = list(zip(pred,confi,*args))
    all = [all[k] for k in range(len(all)) if k not in dropped_junc_index]
    pred, confi, *args = zip(*all)
    return pred,confi,*args


def line_nms(pred,confi,*args,**kwargs):
    # pred: N,2,3
    def line_to_line_dist(xxx):
        # inpu two array  (N0,2,2) 和 (N1,2,2) ; (2,2): [(x1,y1),(x2,y2)]
        diff = ((xxx[:, None, :, None] - xxx[:, None]) ** 2).sum(-1)
        # print(diff.shape)
        diff = np.sqrt(diff)
        diff = np.minimum(
            diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
        )
        return diff
    all = list(zip(pred,confi,*args))
    pred,confi,*args = zip(*sorted(all, reverse=True, key=lambda x:x[1]))

    predArray = np.array(pred,dtype=np.float32)
    dropped_junc_index = []
    nms_threshhold = 5 if "nms_threshhold" not in kwargs.keys() else kwargs['nms_threshhold']
    dist_array = line_to_line_dist(predArray)
    for j in range(len(all)):
        if j in dropped_junc_index:
            continue
        # dist_all = np.linalg.norm(predArray - predArray[j], axis=1)
        dist_all = dist_array[j]
        same_region_indexes = (dist_all < nms_threshhold).nonzero()
        for same_region_index_i in same_region_indexes[0]:
            if same_region_index_i == j:
                continue
            if confi[same_region_index_i] <= confi[j]:
                dropped_junc_index.append(same_region_index_i)
            else:
                dropped_junc_index.append(j)
    all = list(zip(pred,confi,*args))
    all = [all[k] for k in range(len(all)) if k not in dropped_junc_index]
    pred, confi, *args = zip(*all)
    return pred,confi,*args


def confMask(pred,confi,*args):
    thresh=0.5
    all = list(zip(pred, confi, *args))
    all = [all[k] for k in range(len(all)) if all[k][1]>thresh]
    pred, confi, *args = zip(*all)
    return pred,confi,*args

def juncRecall(predXYZ,confi,wireframeJunc,mean,max,fpsPoint):
    # inv-normalize
    predXYZ = (predXYZ+fpsPoint) * max + mean
    wireframeJunc = torch.Tensor(wireframeJunc).to(max.device) * max + mean

    # nms and confi-based mask
    predXYZ = predXYZ[0].tolist()
    confi = confi[0].tolist()
    predXYZ,confi = nms(predXYZ,confi)
    predXYZ,confi = confMask(predXYZ,confi) # prediction after nms and confiMask
    # cal recall
    predXYZ = torch.Tensor(predXYZ).float().to(max.device)
    gtXYZ = wireframeJunc[0]
    N, _ = gtXYZ.shape  # N,3
    S, _ = predXYZ.shape  # S,3
    gtXYZ = gtXYZ.unsqueeze(1).repeat((1, S, 1))  # N,S,3
    positiveXYZ = predXYZ.unsqueeze(0).repeat((N, 1, 1))
    dist = torch.sum((gtXYZ - positiveXYZ) ** 2, dim=-1)  # N,S
    dist = torch.sqrt(dist)
    mindistMask = (torch.min(dist, dim=-1)[0] < 7)
    # disPositive.append(mindistMask.sum() / wireframeJunc.shape[1])
    # print(mindistMask.sum() / wireframeJunc.shape[1])
    return mindistMask.sum() / wireframeJunc.shape[1]


def juncRecallV2(predXYZ,confi,wireframeJunc,mean,max,fpsPoint):
    # inv-normalize
    predXYZ = (predXYZ+fpsPoint) * max + mean
    wireframeJunc = torch.Tensor(wireframeJunc).to(max.device) * max + mean
    fpsPoint = fpsPoint * max + mean
    # nms and confi-based mask
    predXYZ = predXYZ[0].tolist()
    confi = confi[0].tolist()
    predXYZ, confi = nms(predXYZ,confi)

    # cal recall
    predXYZ = torch.Tensor(predXYZ).float().to(max.device)
    gtXYZ = wireframeJunc[0]
    N, _ = gtXYZ.shape  # N,3
    S, _ = predXYZ.shape  # S,3
    gtXYZ = gtXYZ.unsqueeze(1).repeat((1, S, 1))  # N,S,3
    positiveXYZ = predXYZ.unsqueeze(0).repeat((N, 1, 1))
    dist = torch.sum((gtXYZ - positiveXYZ) ** 2, dim=-1)  # N,S
    dist = torch.sqrt(dist).transpose(1,0) # S,N

    choice = torch.argmin(dist,dim=-1)
    dist = torch.min(dist,dim=-1)[0]
    hit = torch.zeros(N,dtype=torch.bool)
    tp = torch.zeros(dist.shape[0],dtype=torch.float64)
    fp = torch.zeros(dist.shape[0], dtype=torch.float64)

    for i in range(dist.shape[0]):
        if dist[i]<7 and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return torch.sum(tp) / wireframeJunc.shape[1]


class getsap:
    def __init__(self, s=7, nms_threshhold=5, confi_thresh=0):
        self.s = s
        self.nms_threshhold = nms_threshhold
        self.tp_list = []
        self.fp_list = []
        self.score_list = []
        self.len=0
        self.use=False
        self.confi_thresh = confi_thresh
        self.pred_point_number = []
        self.pred_point_number_nms = []

    def __call__(self,predXYZ,confi,wireframeJunc,mean,max,fpsPoint,rec_label):
        # inv-norm
        predXYZ = (predXYZ + fpsPoint) * max + mean
        wireframeJunc = torch.Tensor(wireframeJunc).to(max.device) * max + mean
        fpsPoint = fpsPoint * max + mean
        # nms and confi-based mask
        predXYZ = predXYZ[0].tolist()
        confi = confi[0].tolist()

        # truncate
        ALL = list(zip(predXYZ,confi))
        ALL = [kk for kk in ALL if kk[1]>self.confi_thresh]
        predXYZ, confi = zip(*ALL)
        self.pred_point_number.append(len(predXYZ))

        # nms
        predXYZ, confi = nms(predXYZ, confi, nms_threshhold=self.nms_threshhold)
        self.pred_point_number_nms.append(len(predXYZ))

        # cal recall
        predXYZ = torch.Tensor(predXYZ).float().to(max.device)
        gtXYZ = wireframeJunc[0]
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

        self.tp_list.append(tp)
        self.fp_list.append(fp)
        self.score_list.append(torch.Tensor(confi))
        self.len+=N
    def __reset__(self):
        self.tp_list = []
        self.fp_list = []
        self.len = 0
        self.score_list = []

    # ret ap
    def getap(self):
        if self.len==0:
            return 0
        tp_all = torch.cat(self.tp_list)
        fp_all = torch.cat(self.fp_list)
        score_all = torch.cat(self.score_list)
        score_index = torch.argsort(-score_all)
        tp_all = torch.cumsum(tp_all[score_index], dim=0) / self.len
        fp_all = torch.cumsum(fp_all[score_index], dim=0) / self.len
        return self.ap(tp_all, fp_all)

    def get_RPF(self):
        best_F, best_F_confi, best_F_recall,best_F_precision = self.get_best_F()
        tp_all = torch.cat(self.tp_list)
        fp_all = torch.cat(self.fp_list)
        sum_tp = torch.sum(tp_all)
        sum_fp = torch.sum(fp_all)
        recall = sum_tp/self.len
        precision = sum_tp/(sum_fp+sum_tp)
        f1 = 2*recall*precision/(recall+precision)
        AP = self.getap()
        ret = {"AP": AP,"recall": recall.item(),
               "epsilon":self.s,"nms_thresh":self.nms_threshhold,"confi_thresh":self.confi_thresh,
               "pred_min":np.array(self.pred_point_number).min(), "pred_max":np.array(self.pred_point_number).max(),
               "pred_ave":np.array(self.pred_point_number).mean(),"pred_std":np.array(self.pred_point_number).std(),
               "pred_nms_min": np.array(self.pred_point_number_nms).min(), "pred_nms_max": np.array(self.pred_point_number_nms).max(),
               "pred_nms_ave": np.array(self.pred_point_number_nms).mean(), "pred_nms_std": np.array(self.pred_point_number_nms).std(),
               "best_F_recall":best_F_recall,"best_F_precision":best_F_precision}
        ret = {k: round(v, 4) for k, v in ret.items()}
        return ret

    def get_best_F(self):
        if self.len==0:
            return 0
        tp_all = torch.cat(self.tp_list)
        fp_all = torch.cat(self.fp_list)
        score_all = torch.cat(self.score_list)
        score_index = torch.argsort(-score_all)
        tp_all = torch.cumsum(tp_all[score_index],dim=0) / self.len
        fp_all = torch.cumsum(fp_all[score_index],dim=0) / self.len
        recall = tp_all
        precision = tp_all/np.maximum(tp_all + fp_all, 1e-9*torch.ones_like(tp_all))
        F = 2*recall*precision/(recall+precision+1e-9)
        best_F = torch.max(F).item()
        arg_i = torch.argmax(F)
        return best_F,score_all[arg_i].item(),recall[arg_i].item(),precision[arg_i].item()

    # cal recall and precision, -> ret ap
    def ap(self, tp, fp):
        recall = tp
        precision = tp / np.maximum(tp + fp, 1e-9*torch.ones_like(tp)) # tp/P

        recall = torch.cat(( torch.Tensor([0.0]), recall, torch.Tensor([1.0])))  # add begin and end
        precision = torch.cat((torch.Tensor([0.0]), precision, torch.Tensor([0.0])))

        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i]) # down trend
        i = torch.where(recall[1:] != recall[:-1])[0]
        return torch.sum((recall[i + 1] - recall[i]) * precision[i + 1]).item()  # area under PR curve

    def save_result(self,path):
        ret = {
            "tp":torch.cat(self.tp_list).tolist(),
            "fp":torch.cat(self.fp_list).tolist(),
            "gt":self.len,
            "score":torch.cat(self.score_list).tolist(),
        }
        b = json.dumps(ret)
        fw = open(path,'w')
        fw.write(b)
        fw.close()

def getJunc3D(predXYZ, targetXYZ, max, mean, nameAll, labelAll, objGTJunc3DAll, objLineIdxAll, log_path, confiAll, fpsPointAll):
    bs = predXYZ.shape[0]
    # max = max.cpu().detach().numpy()
    # mean = mean.cpu().detach().numpy()
    for i in range(bs):
        pred, target = predXYZ[i], targetXYZ[i]
        fpsPoint = fpsPointAll[i]
        pred = (pred+fpsPoint) * max + mean
        target = (target+fpsPoint) * max + mean

        fpsPoint = fpsPoint * max + mean

        confi,label = confiAll[i],labelAll[i]
        name,objLineIdx,objGTJunc3D = nameAll[i],objLineIdxAll[i],objGTJunc3DAll[i]
        objGTJunc3D = objGTJunc3D * max.cpu().numpy() + mean.cpu().numpy()
        ret={
            "pred":pred.tolist(),
            "target":target.tolist(),
            "confi":confi.tolist(),
            "label":label.tolist(),
            "wireframeLine":objLineIdx.tolist(),
            "wireframeJunc":objGTJunc3D.tolist(),
            "max":max.tolist(),
            "mean":mean.tolist(),
            "fpsPoint":fpsPoint.tolist(),
        }
        b = json.dumps(ret)
        # outDir = "/data/obj_data/vis/visPredJunc"
        outDir = "/data/obj_data/vis/visPredJuncFigure2"
        os.makedirs(outDir,exist_ok=True)
        fw = open(os.path.join(outDir, name + ".json"), 'w')
        fw.write(b)
        fw.close()




#####################valid connectivity########################

class edgeSap:
    def __init__(self,s=7,nms_threshhold=5.0,confi_thresh=0.0):
        self.s=s
        self.nms_threshhold = nms_threshhold
        self.tp_list=[]
        self.fp_list=[]
        self.score_list=[]
        self.len=0
        self.use=False
        self.confi_thresh = confi_thresh
        self.pred_point_number = []
        self.pred_point_number_nms = []
    def __call__(self,predXYZ,combination, edgeConfi,wireframeJunc,wireframeLine, mean,max,fpsPoint,):
        def line_to_line_dist_torch(xx,yy):
            # input two array  (N0,2,2) 和 (N1,2,2) ; (2,2): [(x1,y1),(x2,y2)]
            diff = ((xx[:, None, :, None] - yy[:, None]) ** 2).sum(-1)
            # print(diff.shape)
            diff = torch.sqrt(diff)
            diff = torch.minimum(
                diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
            )
            return diff
        # inv-normalize
        predXYZ = torch.from_numpy(predXYZ[0]).float().to(max.device)
        predXYZ = predXYZ  * max + mean
        wireframeJunc = torch.Tensor(wireframeJunc).to(max.device) * max + mean
        # fpsPoint = fpsPoint * max + mean
        # nms and confi-based mask
        predXYZ = predXYZ[0]
        confi = edgeConfi[0].tolist()

        combineXYZ = predXYZ[combination,:].tolist()

        # truncate
        ALL = list(zip(combineXYZ,confi))
        ALL = [kk for kk in ALL if kk[1]>self.confi_thresh]
        combineXYZ, confi = zip(*ALL)
        self.pred_point_number.append(len(combineXYZ))

        #nms
        combineXYZ, confi = line_nms(combineXYZ, confi, nms_threshhold=self.nms_threshhold)
        self.pred_point_number_nms.append(len(combineXYZ))

        # cal recall
        combineXYZ = torch.Tensor(combineXYZ).float().to(max.device)
        gtCombineXYZ = wireframeJunc[0][wireframeLine,:]
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
        self.tp_list.append(tp)
        self.fp_list.append(fp)
        self.score_list.append(torch.Tensor(confi))
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
        best_F, best_F_confi, best_F_recall,best_F_precision = self.get_best_F()
        tp_all = torch.cat(self.tp_list)
        fp_all = torch.cat(self.fp_list)
        sum_tp = torch.sum(tp_all)
        sum_fp = torch.sum(fp_all)
        recall=sum_tp/self.len
        precision=sum_tp/(sum_fp+sum_tp)
        f1 = 2*recall*precision/(recall+precision)
        AP = self.getap()
        ret = {"AP":AP, "recall":recall.item(), "precision":precision.item(),
               'f1':f1.item(), 'sum_tp':sum_tp.item(), 'sum_fp':sum_fp.item(), 'gt_num':self.len,
               "nms_thresh":self.nms_threshhold, "confi_thresh":self.confi_thresh, "epsilon":self.s,
               "pred_min":np.array(self.pred_point_number).min(), "pred_max":np.array(self.pred_point_number).max(),
               "pred_ave":np.array(self.pred_point_number).mean(),"pred_std":np.array(self.pred_point_number).std(),
               "pred_nms_min": np.array(self.pred_point_number_nms).min(), "pred_nms_max": np.array(self.pred_point_number_nms).max(),
               "pred_nms_ave": np.array(self.pred_point_number_nms).mean(), "pred_nms_std": np.array(self.pred_point_number_nms).std(),
               "best_F":best_F, "best_F_confi":best_F_confi,
               "best_F_recall":best_F_recall,"best_F_precision":best_F_precision}

        return ret

    def get_best_F(self):
        if self.len==0:
            return 0
        tp_all = torch.cat(self.tp_list)
        fp_all = torch.cat(self.fp_list)
        score_all = torch.cat(self.score_list)
        score_index = torch.argsort(-score_all)
        tp_all = torch.cumsum(tp_all[score_index],dim=0) / self.len
        fp_all = torch.cumsum(fp_all[score_index],dim=0) / self.len
        recall = tp_all
        precision = tp_all/np.maximum(tp_all + fp_all, 1e-9*torch.ones_like(tp_all))
        F = 2*recall*precision/(recall+precision+1e-9)
        best_F = torch.max(F).item()
        arg_i = torch.argmax(F)
        return best_F,score_all[arg_i].item(),recall[arg_i].item(),precision[arg_i].item()

    def ap(self,tp, fp): # calulate recall and precision,
        recall = tp
        precision = tp / np.maximum(tp + fp, 1e-9*torch.ones_like(tp)) # tp/P

        recall = torch.cat(( torch.Tensor([0.0]), recall, torch.Tensor([1.0]) ))
        precision = torch.cat((torch.Tensor([0.0]), precision, torch.Tensor([0.0])))

        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i]) #
        i = torch.where(recall[1:] != recall[:-1])[0]
        return torch.sum((recall[i + 1] - recall[i]) * precision[i + 1]).item()  # area under PR curve

    def print_result(self):
        d ={}
        d['Sap{}'.format(str(self.s))]=self.getap()
        adict = self.get_RPF()
        out = {**d,**adict}
        print(out)

    def save_result(self,path):
        ret = {
            "tp":torch.cat(self.tp_list).tolist(),
            "fp":torch.cat(self.fp_list).tolist(),
            "gt":self.len,
            "score":torch.cat(self.score_list).tolist(),
        }
        b = json.dumps(ret)
        fw = open(path,'w')
        fw.write(b)
        fw.close()


def Acc(logits,target):
    prob = F.softmax(logits, dim=1)
    pred = torch.argmax(prob, dim=1)
    acc = (pred==target).sum()
    return acc/pred.shape[1]

def recallAndAcc(logits,target):
    prob = F.softmax(logits, dim=1)
    # pred = torch.argmax(prob, dim=1)
    pred = prob[:,1,:]>0.2 #
    tp1 = ((pred==1)*(target==1))
    # tp1 = acc * (target==1)
    recall_1 = tp1.sum()/(target==1).sum()
    precision = tp1.sum()/((pred==1).sum()+1e-8)
    return recall_1,precision



def fun2(predXYZ, label, combine, logits, wireframeJunc, wireframeLine, name, mean, std, predXYZconfi, outputdir):
    ret={
        "mean":mean.cpu().numpy().tolist(),
        "std":std.cpu().numpy().tolist(),
        "predXYZ": predXYZ.tolist(),
        "predXYZconfi":predXYZconfi[0].tolist(),
        "label": label.tolist(),
        "combine":combine.tolist(),
        "prob": F.softmax(logits,dim=1)[0].cpu().numpy().tolist(),
        "wireframeJunc":wireframeJunc.tolist(),
        "wireframeLine":wireframeLine.tolist(),
    }
    b = json.dumps(ret)
    dir = outputdir
    os.makedirs(dir,exist_ok=True)
    fw = open(os.path.join(dir,name+".json"), 'w')
    fw.write(b)
    fw.close()

def fun2V2_forFigure2(predXYZ, label, combine, logits, wireframeJunc, wireframeLine, name, mean, std, predXYZconfi):
    # 先merge再连
    predXYZ = torch.from_numpy(predXYZ[0]).to(std.device)
    predXYZ = predXYZ * std + mean.unsqueeze(0)
    ret={
        "mean":mean.cpu().numpy().tolist(),
        "std":std.cpu().numpy().tolist(),
        "predXYZ": predXYZ.tolist(),
        "predXYZconfi":predXYZconfi[0].tolist(),
        "label": label.tolist(),
        "combine":combine.tolist(),
        "prob": F.softmax(logits,dim=1)[0].cpu().numpy().tolist(),
        "connect_prob":F.softmax(logits,dim=1)[0][1].cpu().numpy().tolist(),
        "wireframeJunc":wireframeJunc.tolist(),
        "wireframeLine":wireframeLine.tolist(),
    }
    b = json.dumps(ret)
    dir = os.path.join("/data/obj_data/vis","visPredConnectFigure2")
    os.makedirs(dir,exist_ok=True)
    fw = open(os.path.join(dir,name+"predConnect_confiMask.json"), 'w')
    fw.write(b)
    fw.close()

import time
def validCls(model, data_loader, log_path, cfg):
    model.eval()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(1), max=2)
    avg_loss_stats = None
    CONFI_THRESH = 0.001
    S10_0 = edgeSap(s=10,nms_threshhold=0.01,confi_thresh=CONFI_THRESH)
    S7_0 = edgeSap(s=7, nms_threshhold=0.01,confi_thresh=CONFI_THRESH)
    S5_0 = edgeSap(s=5,nms_threshhold=0.01,confi_thresh=CONFI_THRESH)

    S10 = edgeSap(s=10,nms_threshhold=4,confi_thresh=CONFI_THRESH)
    S7 = edgeSap(s=7, nms_threshhold=4,confi_thresh=CONFI_THRESH)
    S5 = edgeSap(s=5,nms_threshhold=4,confi_thresh=CONFI_THRESH)



    tAll = 0
    tstart = time.time()
    with torch.no_grad():
        for iter_id, batch in enumerate(data_loader):

            bs = batch['input'].size(0)
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda()
            t1 = time.time()
            output = model(batch)
            tAll = tAll + time.time() - t1

            logits = output['l1_logits']



            label,classifyLabel = batch['label'],batch['classifyLabel']
            N = classifyLabel.shape[1]
            fpsPoint, predXYZ = batch['fpsPoint'], batch['predXYZ']
            combine, wireframeLine = batch['combine'][0], batch['objLineIdx'][0]
            acc = Acc(logits, batch['classifyLabel'])
            recall,precision = recallAndAcc(logits, batch['classifyLabel'])

            confi = F.softmax(logits,dim=1)[:,1,:]

            S10(batch['predXYZ'],combine, confi,batch['objGTJunc3D'],wireframeLine, batch['mean'],batch['std'],batch['fpsPoint'],)
            S7(batch['predXYZ'], combine, confi, batch['objGTJunc3D'], wireframeLine, batch['mean'], batch['std'],batch['fpsPoint'])
            S5(batch['predXYZ'], combine, confi, batch['objGTJunc3D'], wireframeLine, batch['mean'], batch['std'],batch['fpsPoint'])

            S10_0(batch['predXYZ'],combine, confi,batch['objGTJunc3D'],wireframeLine, batch['mean'],batch['std'],batch['fpsPoint'],)
            S7_0(batch['predXYZ'], combine, confi, batch['objGTJunc3D'], wireframeLine, batch['mean'], batch['std'],batch['fpsPoint'])
            S5_0(batch['predXYZ'], combine, confi, batch['objGTJunc3D'], wireframeLine, batch['mean'], batch['std'],batch['fpsPoint'])

            if cfg['write_pred_wireframe']:
                fun2(predXYZ[0], label[0], combine, logits, batch['objGTJunc3D'][0], wireframeLine, batch['name'][0],batch['mean'][0],batch['std'],batch['predXYZconfi'],cfg['output_path'])

            metric={
                'cls_acc_loss':acc,
                'recall_loss':recall,
                'pre_loss':precision,
            }
            if avg_loss_stats is None:
                avg_loss_stats = {l: AverageMeter() for l in metric.keys() if "loss" in l}
            Bar.suffix = '[{0}/{1}]|'.format(iter_id, num_iters)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(metric[l].mean().item(), bs)
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if iter_id%50==0: print(Bar.suffix)

        def get_avg(*args):
            AP_all = 0
            recall_all = 0
            for a in args:
                AP_all+=a['AP']
                recall_all+=a['recall']
            AP_all = AP_all/len(args)
            recall_all = recall_all / len(args)
            return round(AP_all, 4), round(recall_all, 4)

        S10_result = S10.get_RPF()
        S7_result = S7.get_RPF()
        S5_result = S5.get_RPF()

        S10_0_result = S10_0.get_RPF()
        S7_0_result = S7_0.get_RPF()
        S5_0_result = S5_0.get_RPF()
        avg_ap, avg_recall = get_avg(S5_0_result,S7_0_result,S10_0_result)
        print("""
            vtx min/max/ave/std {},{},{},{}
            sap5, recall5, {}, {},
            sap7, recall7, {}, {},
            sap10, recall10, {}, {},
            ave sap, ave recall, {},{}

            """.format(S10_0_result['pred_min'], S10_0_result['pred_max'],
                       S10_0_result['pred_ave'], S10_0_result['pred_std'],
                       S5_0_result['AP'], S5_0_result['recall'],
                       S7_0_result['AP'], S7_0_result['recall'],
                       S10_0_result['AP'], S10_0_result['recall'],
                       avg_ap, avg_recall,
                       )
              )

        print("network forward total time {}, total sample {}, ave {}".format(tAll,data_loader.__len__(),tAll/data_loader.__len__()))
        FF = time.time() - tstart
        print("network total time {}, total sample {}, ave {}".format(FF, data_loader.__len__(),FF / data_loader.__len__()))
        print("save tp fp result of nms=3")
        return S5_0_result['AP']