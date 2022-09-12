import numpy as np
import glob
import cv2
import scipy.io as sio
import torch
import json
'''
ap sap metric
DATE: 2022-03-16

'''
def line_nms(pred,confi,*args,**kwargs):
    # pred: N,2,3
    def line_to_line_dist(xxx):
        # input two array  (N0,2,2) 和 (N1,2,2) ; (2,2): [(x1,y1),(x2,y2)]
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


def msTPFP(line_pred, line_gt, threshold):
    # input two array  (N0,2,2) 和 (N1,2,2) ; (2,2): [(x1,y1),(x2,y2)]
    # print(line_pred[:,None,:,None].shape,line_gt[:,None].shape)
    diff = ((line_pred[:,None,:] - line_gt[None,:,:]) ** 2).sum(-1)


    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), np.bool_) # record if used
    tp = np.zeros(len(line_pred), np.float64)
    fp = np.zeros(len(line_pred), np.float64)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
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
        ret = {"recall":recall,"precision":precision,'f1':f1,'sum_tp':sum_tp,'sum_fp':sum_fp,'gt_num':self.len}
        return ret
        return recall,precision,f1,sum_tp,sum_fp,self.len
    def ap(self,tp, fp):
        recall = tp
        precision = tp / np.maximum(tp + fp, 1e-9*torch.ones_like(tp)) # tp/P

        recall = torch.cat(( torch.Tensor([0.0]), recall, torch.Tensor([1.0]) ))
        precision = torch.cat((torch.Tensor([0.0]), precision, torch.Tensor([0.0])))

        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        i = torch.where(recall[1:] != recall[:-1])[0]
        return torch.sum((recall[i + 1] - recall[i]) * precision[i + 1])

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
            diff = ((xx[:, None, :, None] - yy[:, None]) ** 2).sum(-1)
            diff = torch.sqrt(diff)
            diff = torch.minimum(
                diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
            )
            return diff
        # inv-normalize
        predXYZ = torch.Tensor(predXYZ).float()

        wireframeJunc = torch.Tensor(wireframeJunc)


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

    def ap(self,tp, fp):
        recall = tp
        precision = tp / np.maximum(tp + fp, 1e-9*torch.ones_like(tp)) # tp/P

        recall = torch.cat(( torch.Tensor([0.0]), recall, torch.Tensor([1.0]) ))
        precision = torch.cat((torch.Tensor([0.0]), precision, torch.Tensor([0.0])))

        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        i = torch.where(recall[1:] != recall[:-1])[0]
        return torch.sum((recall[i + 1] - recall[i]) * precision[i + 1])

    def print_result(self):
        d ={}
        d['Sap{}'.format(str(self.s))]=self.getap()
        adict = self.get_RPF()
        out = {**d,**adict}
        print(out)


class edgeSap:
    def __init__(self,s=7,nms_threshhold=0,confi_thresh=0):
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
    def __call__(self,predXYZ,combination, edgeConfi,wireframeJunc,wireframeLine):
        def line_to_line_dist_torch(xx,yy):
            diff = ((xx[:, None, :, None] - yy[:, None]) ** 2).sum(-1)
            # print(diff.shape)
            diff = torch.sqrt(diff)
            diff = torch.minimum(
                diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
            )
            return diff
        # 反归一化

        predXYZ = torch.Tensor(predXYZ).float()
        wireframeJunc = torch.from_numpy(wireframeJunc)

        ALL = list(zip(combination,edgeConfi))
        ALL = [kk for kk in ALL if kk[1]>self.confi_thresh]
        ALL = sorted(ALL,key=lambda x:x[1],reverse=True)
        combination,edgeConfi = zip(*ALL)
        self.pred_point_number.append(len(combination))  
        # nms and confi-based mask
    
        confi = edgeConfi
        combineXYZ = predXYZ[combination,:].tolist()
        combineXYZ, confi = line_nms(combineXYZ, confi, nms_threshhold=self.nms_threshhold)
        self.pred_point_number_nms.append(len(combineXYZ))
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
        ret = {"AP":AP,"recall":recall.item(),
               "precision":precision.item(),'f1':f1.item(),'sum_tp':sum_tp.item(),'sum_fp':sum_fp.item(),'gt_num':self.len,
               "epsilon":self.s,"nms_thresh":self.nms_threshhold,"confi_thresh":self.confi_thresh,
               "pred_min":np.array(self.pred_point_number).min(), "pred_max":np.array(self.pred_point_number).max(), "pred_ave":np.array(self.pred_point_number).mean(),"pred_std":np.array(self.pred_point_number).std(),
               "pred_nms_min": np.array(self.pred_point_number_nms).min(), "pred_nms_max": np.array(self.pred_point_number_nms).max(),"pred_nms_ave": np.array(self.pred_point_number_nms).mean(), "pred_nms_std": np.array(self.pred_point_number_nms).std(),
               #"best_F":best_F,"best_F_confi":best_F_confi,
               "best_F_recall":best_F_recall,"best_F_precision":best_F_precision,
               "pred_stat":[np.array(self.pred_point_number).min(), np.array(self.pred_point_number).max(), np.array(self.pred_point_number).mean(),np.array(self.pred_point_number).std()],
               }

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
    def ap(self,tp, fp):
        recall = tp
        precision = tp / np.maximum(tp + fp, 1e-9*torch.ones_like(tp)) # tp/P

        recall = torch.cat(( torch.Tensor([0.0]), recall, torch.Tensor([1.0]) ))
        precision = torch.cat((torch.Tensor([0.0]), precision, torch.Tensor([0.0])))

        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        i = torch.where(recall[1:] != recall[:-1])[0]
        return torch.sum((recall[i + 1] - recall[i]) * precision[i + 1])
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

class getsap:
    def __init__(self,s=7,nms_threshhold=5,  confi_thresh=0):
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

    def __call__(self,predXYZ,confi,wireframeJunc):
        predXYZ = predXYZ.tolist()
        confi = confi.tolist()
        # truncate
        ALL = list(zip(predXYZ,confi))
        ALL = [kk for kk in ALL if kk[1]>self.confi_thresh]
        ALL = sorted(ALL,key=lambda x:x[1],reverse=True)
        predXYZ, confi = zip(*ALL)
        self.pred_point_number.append(len(predXYZ))


        # predXYZ, confi = nms(predXYZ, confi, nms_threshhold=self.nms_threshhold)
        self.pred_point_number_nms.append(len(predXYZ))

        # cal recall
        predXYZ = torch.Tensor(predXYZ).float()
        gtXYZ = torch.from_numpy(wireframeJunc).float()
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
        ret = {"AP":AP,"recall":recall.item(),
               "precision":precision.item(),'f1':f1.item(),'sum_tp':sum_tp.item(),'sum_fp':sum_fp.item(),'gt_num':self.len,
               "epsilon":self.s,"nms_thresh":self.nms_threshhold,"confi_thresh":self.confi_thresh,
               "pred_min":np.array(self.pred_point_number).min(), "pred_max":np.array(self.pred_point_number).max(), "pred_ave":np.array(self.pred_point_number).mean(),"pred_std":np.array(self.pred_point_number).std(),
               "pred_nms_min": np.array(self.pred_point_number_nms).min(), "pred_nms_max": np.array(self.pred_point_number_nms).max(),"pred_nms_ave": np.array(self.pred_point_number_nms).mean(), "pred_nms_std": np.array(self.pred_point_number_nms).std(),
               #"best_F":best_F,"best_F_confi":best_F_confi,
               "pred_stat":[np.array(self.pred_point_number).min(), np.array(self.pred_point_number).max(), np.array(self.pred_point_number).mean(),np.array(self.pred_point_number).std()],
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
        

    def ap(self,tp, fp):
        recall = tp
        precision = tp / np.maximum(tp + fp, 1e-9*torch.ones_like(tp)) # tp/P

        recall = torch.cat(( torch.Tensor([0.0]), recall, torch.Tensor([1.0]) ))
        precision = torch.cat((torch.Tensor([0.0]), precision, torch.Tensor([0.0])))

        for i in range(precision.shape[0] - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        i = torch.where(recall[1:] != recall[:-1])[0]
        return torch.sum((recall[i + 1] - recall[i]) * precision[i + 1])


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
