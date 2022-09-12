import torch.utils.data as data
import torch
import numpy as np
import os
import json
from itertools import permutations,combinations
import torch.nn.functional as F
from sklearn.decomposition import PCA
import modeling.utils as MUT
import random


def cmp(x,y):
    for i in range(3):
        if x[i]<y[i]:
            return -1
        elif x[i]>y[i]:
            return 1
        else:
            continue
    return 0


def readWireframeGT(name):

    def face_to_line(face_idx):
        line_idx=[]
        for f in face_idx:
            # f = f + [f[0]]
            for vf in range(len(f)):
                t = [ f[vf], f[(vf+1)%len(f)] ]
                t = sorted(t)
                if t not in line_idx:
                    line_idx.append(t)
        return line_idx
    point=[]
    face_idx=[]
    with open(name, 'r') as fr:
        while True:
            line = fr.readline()[:-1]
            if not line:
                break
            else:
                x = line.split()
                if x[0] == 'v':
                    p3D = [float(i) for i in x[1:]]
                    p3D[1],p3D[2] = -p3D[2], p3D[1]
                    point.append(p3D)
                elif x[0] == 'f':
                    lIdx = [int(i.split('/')[0])-1 for i in x[1:]]
                    face_idx.append(lIdx)
                else:
                    continue
    line_idx = face_to_line(face_idx)
    objGTJunc3D = np.array(point,dtype=np.float64)
    objGTSeg3D = []
    for e in line_idx:
        objGTSeg3D.append(point[e[0]]+point[e[1]])
    objGTSeg3D = np.array(objGTSeg3D, dtype=np.float64)
    return objGTJunc3D, objGTSeg3D, line_idx


def checkPointV2(fpsPoints, label, wireframeGT, LineIdx, name):
    # check label recall under fps sample
    ret = {
        "fpsPoints": fpsPoints,
        "label": label,
        "wireframeGT":wireframeGT,
        "wireframeLine":LineIdx,
    }
    b = json.dumps(ret)
    fw = open(os.path.join("/data/obj_data","tempResults",name+".json"), 'w')
    fw.write(b)
    fw.close()


def invNormalize(points, max, mean):
    return (points * max) + mean


def writeTwoGroupLine(seg3D, wireframeJunc, edge, name):
    out = {
        "seg3D":seg3D.cpu().detach().numpy().tolist(),
        "wireframeJunc":wireframeJunc.tolist(),
        "edge":edge,
    }
    b = json.dumps(out)
    fw = open(os.path.join("/data/obj_data","twogroupJson",name+".json"), 'w')
    fw.write(b)
    fw.close()


def visDynamicStep2(predXYZ, predLabel, wireframeJunc, wireframeLine, edge, label, name):
    out = {
        "predXYZ":predXYZ,
        "predLabel":predLabel,
        "wireframeJunc":wireframeJunc,
        "wireframeLine":wireframeLine,
        "edge":edge, # connectivity of pred edge
        "label":label,
    }
    b = json.dumps(out)
    fw = open(os.path.join("/data/obj_data","step2",name+".json"), 'w')
    fw.write(b)
    fw.close()

def visDynamicStep3(predXYZ, wireframeJunc, wireframeLine, edge, label, fpsPoint,name):
    out = {
        "predXYZ":predXYZ,
        "wireframeJunc":wireframeJunc,
        "wireframeLine":wireframeLine,
        "edge":edge, # connectivity of pred edge
        "fpsPoint":fpsPoint,
        "label":label,
    }
    outPath = "/data/obj_data/vis/visStep3Figure2"
    os.makedirs(outPath,exist_ok=True)
    b = json.dumps(out)
    fw = open(os.path.join(outPath,name+".json"), 'w')
    fw.write(b)
    fw.close()

def visDynamicStep3_Figure2(predXYZ, wireframeJunc, wireframeLine, edge, label, fpsPoint,name,mean,std,combine_label,outPath=None):
    #画图2时的可视化
    groupLines = torch.Tensor(predXYZ)
    groupLines = groupLines * std
    groupLines = groupLines + mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    fpsPoint = torch.Tensor(fpsPoint)
    fpsPoint = fpsPoint * std + mean.unsqueeze(0)
    fpsPoint = fpsPoint.view(-1,2,3)
    out = {
        "predXYZ":predXYZ,
        "wireframeJunc":wireframeJunc,
        "wireframeLine":wireframeLine,
        "edge":edge, # connectivity of pred edge
        "fpsPoint":fpsPoint,
        "label":label,
    }
    # outPath = "/data/obj_data/vis/visStep3Figure2"
    if outPath is None:
        outPath = "/data/obj_data/vis/visStep3Figure2_sampleLineALL"
    outPath = os.path.join(outPath,name)
    os.makedirs(outPath,exist_ok=True)
    for i in range(groupLines.shape[0]):
        vert = groupLines[i].view(-1,3).tolist()
        connect = np.arange(vert.__len__()).reshape(-1,2).tolist()
        endpoint = fpsPoint[i].tolist()
        name = "{}_c{}___{}vs{}.obj".format(str(i),str(label[i]),str(combine_label[i][0]),str(combine_label[i][1]))
        with open(os.path.join(outPath,name),'w') as fw:
            for v in vert:
                fw.write("v "+ " ".join([str(kk) for kk in v])+"\n")
            for ee in endpoint:
                fw.write("v " + " ".join([str(kk) for kk in ee]) + "\n")
            for l1,l2 in connect:
                fw.write("l {} {}\n".format(l1+1,l2+1))


    # b = json.dumps(out)
    # fw = open(os.path.join(outPath,name+".json"), 'w')
    # fw.write(b)
    # fw.close()


def writeGroupJson(groupJunc3D, groupMylabel, groupSegLabel, groupJuncLabel, wireframeJunc, wireframeLine, fpsLabel, fpsPoint, fpsPointIdx, name, srcJunc, srcJuncLabel):
    groupJunc3D = groupJunc3D.tolist()
    groupMylabel = groupMylabel.tolist()
    ret={
        "groupJunc3D":groupJunc3D, # 256,32,3
        "groupJuncLabel":groupJuncLabel.tolist(), # 256,16,2
        "groupSegLabel":groupSegLabel.tolist(), # 256,16
        "groupMylabel":groupMylabel, #256
        "fpsLabel":fpsLabel.tolist(), #256
        "fpsPoint":fpsPoint.tolist(),
        "fpsPointIdx":fpsPointIdx.tolist(),
        "wireframeJunc":wireframeJunc,
        "wireframeLine":wireframeLine,
        "srcJunc":srcJunc,
        "srcLabel":srcJuncLabel,
    }
    b = json.dumps(ret)
    outDir = "/data/obj_data/vis/visSampleJunc"
    os.makedirs(outDir,exist_ok=True)
    fw = open(os.path.join(outDir,name+".json"), 'w')
    fw.write(b)
    fw.close()


def writeGroupJsonForFigure2(groupJunc3D, groupMylabel, groupSegLabel, groupJuncLabel, wireframeJunc, wireframeLine, fpsLabel, fpsPoint, fpsPointIdx, name, srcJunc, srcJuncLabel):
    groupJunc3D = groupJunc3D.tolist()
    groupMylabel = groupMylabel.tolist()
    ret={
        "groupJunc3D":groupJunc3D, # 256,32,3
        "groupJuncLabel":groupJuncLabel.tolist(), # 256,16,2
        "groupSegLabel":groupSegLabel.tolist(), # 256,16
        "groupMylabel":groupMylabel, #256
        "fpsLabel":fpsLabel.tolist(), #256
        "fpsPoint":fpsPoint.tolist(),
        "fpsPointIdx":fpsPointIdx.tolist(),
        "wireframeJunc":wireframeJunc,
        "wireframeLine":wireframeLine,
        "srcJunc":srcJunc,
        "srcLabel":srcJuncLabel,
    }
    temp = np.array(groupMylabel)
    temp[temp>=0] = 1
    temp[temp<0] = 0
    outDir = "/data/obj_data/vis/visFigure2_allSamplePatch_0221temp"
    outDir = "/data/obj_data/vis/real_temp"
    outDir = os.path.join(outDir,name)
    os.makedirs(outDir,exist_ok=True)
    fpsPoint = fpsPoint[0].tolist()
    for i in range(len(groupJunc3D)):
        # outFileName = str(i)+""+str(temp[i])+".obj"
        outFileName = "{}_{}.obj".format(str(i),str(int(temp[i])))
        vert = groupJunc3D[i]
        connect = np.arange(groupJunc3D[i].__len__()).reshape(-1,2).tolist()
        with open(os.path.join(outDir,outFileName),'w') as fw:
            for v in vert:
                fw.write("v "+" ".join([str(kk) for kk in v]) + "\n")

            fw.write("v "+" ".join([str(kk) for kk in fpsPoint[i]]) + "\n")

            for l1,l2 in connect:
                fw.write("l {} {}\n".format(l1+1,l2+1))



def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def dataAug(*args, scale_low=0.8, scale_high=1.25, shift_range=0.1):
    '''
    :param args: N*C
    :return:
    '''
    scale = np.random.uniform(scale_low, scale_high)
    shift = np.random.uniform(-shift_range, shift_range, (3,))
    ret = [a*scale+shift for a in args]
    return ret

class postProcess:
    def __init__(self,thresh=0.5,nms_threshhold=5):
        self.thresh=thresh
        self.nms_threshhold=nms_threshhold
        self.disThresh=7 # dis threshold of pred Junc matching with Gt Junc
    def nms(self,pred,confi,*args):
        all = list(zip(pred, confi, *args))
        pred, confi, *args = zip(*sorted(all, reverse=True, key=lambda x: x[1]))

        predArray = np.array(pred, dtype=np.float64)
        dropped_junc_index = []
        nms_threshhold = self.nms_threshhold
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
        all = list(zip(pred, confi, *args))
        all = [all[k] for k in range(len(all)) if k not in dropped_junc_index]
        pred, confi, *args = zip(*all)
        return pred, confi, *args

    def confiMask(self,pred,confi,*args):
        thresh = self.thresh
        all = list(zip(pred, confi, *args))
        all = [all[k] for k in range(len(all)) if all[k][1] > thresh]
        pred, confi, *args = zip(*all)
        return pred, confi, *args

    def assignLabel(self,pred,wireframejunc,confi):
        pred, wireframejunc = torch.Tensor(pred).float(), torch.Tensor(wireframejunc).float() # (N,3), (L,3)
        confi = torch.Tensor(confi).float()
        dist = pred[:,None,:] - wireframejunc[None,:,:]
        sqr_dist = torch.sum(dist**2,dim=-1) # N,L
        minSqrDist,minIndex = torch.min(sqr_dist,dim=-1) # (N,)
        mask = (minSqrDist<self.disThresh*self.disThresh)
        selectedInd = mask.nonzero().squeeze(-1)

        label = -1 * torch.ones_like(mask)
        label[selectedInd] = minIndex[selectedInd]
        # selectedPred,selectedLabel,selectedConfi = pred[selectedInd],minIndex[selectedInd],confi[selectedInd]
        return pred,label,confi


class LineDataset(data.Dataset):
    def __init__(self,param=None,split="train"):
        super(LineDataset, self).__init__()
        self.split = split
        self.root_path = param['LineCloud_path']

        self.fpsPointNumber = 256
        self.EachGroupNumber = param['EachGroupNumber']
        self.NegSampleThresh = param['NegSampleThresh']

        self.query_dist = "ptl" # ptl: point to line , ptls: point to line segment
        self.ret_dist = "ptls"
        self.use_real = param['use_real']

        self.temp_fpsPoint_path = self.root_path + "_fps" + "_" + str(self.fpsPointNumber)

        os.makedirs(self.temp_fpsPoint_path,exist_ok=True)

        if split=="train" or split=="valid":
            x = param['train_path']
            with open(x, 'r') as fr:
                readlines = fr.readlines()
                self.obj_list = [os.path.join(self.root_path,i[:-1]+".json") for i in readlines]
                self.fps_list = [os.path.join(self.temp_fpsPoint_path,i[:-1]+".json") for i in readlines]

        else:
            x = param['test_path']
            with open(x, 'r') as fr:
                readlines = fr.readlines()
                self.obj_list = [os.path.join(self.root_path,i[:-1]+".json") for i in readlines]
                self.fps_list = [os.path.join(self.temp_fpsPoint_path, i[:-1] + ".json") for i in readlines]
        print('==> initializing  {} data.'.format(split))
        print('Loaded {} {} samples'.format(len(self.obj_list), split))
        self.postprocess = postProcess()
        self.temp=[]
        self.mean = [0.12249484, -0.3440005,  70.5140106]
        self.std = 266.2135032067564
        self.if_cat = param['if_cat']
        self.cat_normal = param['cat_normal']
        self.normal_patch_line = param['normal_patch_line']
    from ._LineMethodTrans import LineGetItemV3TransOneNormalV2

    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, item):
        return self.LineGetItemV3TransOneNormalV2(item)


    @staticmethod
    def collate_fn(batch):
        X, Mylabel,classifyLabel, objGTJunc3D, fpsPoint, mean, std, name, objLineIdx,rec_label,word_mask,fpsLabel = zip(*batch)
        item = list(range(len(X)))
        ret =  {
            "item": torch.Tensor(item).long(),
            "input": torch.vstack(X),
            "fpsPoint": torch.vstack(fpsPoint),
            "label": torch.vstack(Mylabel).long(),  # 16,256
            "classifyLabel":torch.vstack(classifyLabel),
            "objGTJunc3D": objGTJunc3D,
            "mean":torch.Tensor(mean).float(),
            "std":torch.Tensor(std).float(),
            "name":name,
            "objLineIdx":objLineIdx,
            "rec_label":rec_label,
            "word_mask":torch.vstack(word_mask),
            "fpsLabel":torch.vstack(fpsLabel).long(),
        }
        return ret


class classifyDataset(data.Dataset):
    def __init__(self,param=None,split="train"):
        super(classifyDataset, self).__init__()
        self.split = split
        self.root_path = param['LineCloud_path']
        self.PredJunc_path = param['PredJunc_path']
        self.use_real = param['use_real']

        if split=="train":
            x = param['train_path']
            with open(x, 'r') as fr:
                readlines = fr.readlines()
                self.obj_list = [os.path.join(self.root_path,i[:-1]+".json") for i in readlines]
                self.predJunc_list = [os.path.join(self.PredJunc_path,i[:-1]+".json") for i in readlines]
        else:
            x = param['test_path']
            with open(x, 'r') as fr:
                readlines = fr.readlines()
                self.obj_list = [os.path.join(self.root_path,i[:-1]+".json") for i in readlines]
                self.predJunc_list = [os.path.join(self.PredJunc_path,i[:-1]+".json") for i in readlines]

        print('==> initializing  {} data.'.format(split))
        print('Loaded {} {} samples'.format(len(self.obj_list), split))
        self.postprocess = postProcess(thresh=0.85)

        self.mean = [0.12249484, -0.3440005,  70.5140106]
        self.std = 266.2135032067564
        self.if_cat = param['if_cat']
        self.sample_number = param['sample_number']
    from  ._ClassifyMethod import ClassifyLineStaticAndDynamic,ClassifyLineTestDynamicForTest
    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, item):
        if self.split=='train':
            return self.ClassifyLineStaticAndDynamic(item)
        else:
            return self.ClassifyLineTestDynamicForTest(item)



    @staticmethod
    def collate_fn(batch):
        X, label,classifyLabel, objGTJunc3D, fpsPoint, mean, std, name, combine, predXYZconfi,predXYZ,objLineIdx, word_mask, edgeMask = zip(*batch)
        # X, Mylabel, classifyLabel, objGTJunc3D, fpsPoint, mean, std, name, objLineIdx = zip(*batch)
        item = list(range(len(X)))
        ret =  {
            "item":item,
            "input": torch.vstack(X),
            "fpsPoint": torch.vstack(fpsPoint),
            "label": label, # pred and gt match
            "classifyLabel":torch.vstack(classifyLabel),
            "objGTJunc3D": objGTJunc3D,
            "mean":torch.Tensor(mean).float(),
            "std":torch.Tensor(std).float(),
            "name":name,
            "objLineIdx":objLineIdx,
            "predXYZconfi":predXYZconfi,  # torch.vstack(predXYZconfi),
            "predXYZ":predXYZ,
            "combine":combine,
            "word_mask":torch.vstack(word_mask),
            "edge_mask": torch.Tensor(edgeMask)
            # "heatMap": torch.stack(heatMap,dim=0),
            # "mask": torch.stack(heatMap, dim=0)*9 + 1,
        }
        return ret