import  torch.utils.data as data
import torch
import numpy as np
import os
import functools
import json
from modeling.utils import *
from itertools import permutations,combinations
import torch.nn.functional as F
from sklearn.decomposition import PCA
import modeling.utils as MUT
import random
from dataset import *



def LineGetItemV3TransOneNormalV2(self, item):
    '''

    '''
    json_path = self.obj_list[item]
    name = os.path.split(json_path)[-1][:-5]
    data = json.load(open(json_path, 'r'))

    # data from Line3Dpp
    junc3DList = data['junc3DList']
    seg3DIdx = torch.from_numpy(np.arange(len(junc3DList)).reshape(-1, 2)).long()
    label = data['label']
    rec_label = np.unique(np.array(label))[1:]
    # data from wireframe GT
    objGTJunc3D = data['objGTJunc3D']
    objLineIdx = data['line_idx']

    # normalize
    junc3D_array = np.array(junc3DList, dtype=np.float64)
    if os.path.exists(self.fps_list[item]):
        fpsPointIdx = torch.Tensor(preprocessV2(junc3D_array)).unsqueeze(0).long()
    else:
        fpsPointIdx = torch.Tensor(preprocessV2(junc3D_array)).unsqueeze(0).long()

    mean = np.array(self.mean, dtype=np.float64)
    junc3D_array = junc3D_array - mean[None, :]
    max = self.std
    junc3D_array = junc3D_array / max
    objGTJunc3D_array = np.array(objGTJunc3D, dtype=np.float64)
    objGTJunc3D_array = (objGTJunc3D_array - mean[None, :]) / max

    # scale and shift
    if self.split == "train":
        junc3D_array, objGTJunc3D_array = dataAug(junc3D_array, objGTJunc3D_array)

    junc3D = torch.from_numpy(junc3D_array).float().unsqueeze(0)
    seg3D = junc3D.view(junc3D.shape[0], -1, 2, 3)  # (B,N,2,3)
    fpsPointIdx = farthest_point_sampleV2(junc3D, self.fpsPointNumber, fpsPointIdx)
    if not os.path.exists(self.fps_list[item]):
        b = json.dumps({"fpsPointIdx": fpsPointIdx.numpy().tolist()})
        fw = open(self.fps_list[item], 'w')
        fw.write(b)
        fw.close()
    fpsPoint = index_points(junc3D, fpsPointIdx)
    fpsLabel = torch.Tensor(label)[fpsPointIdx[0]]
    # print(torch.unique(fpsLabel).shape[0]/np.unique(np.array(label)).shape[0], (np.unique(np.array(label)).shape[0]-1)/len(objGTJunc3D))

    EachGroupNumber = 32  # 32
    NegSampleThresh = 16  # 16
    selectedSegIdx, ballMask, sorted_dists, word_mask = query_ball_pointV5(0.2, EachGroupNumber, seg3D,
                                                                           fpsPoint)  # segIdx: (1,256,16)
    toCat = sorted_dists.unsqueeze(-1)

    _, sn, gn = selectedSegIdx.shape
    selectedSegLabel = torch.Tensor(label).view(-1, 2)[selectedSegIdx, :][0]  # 256,16,2
    bs_view = torch.zeros_like(selectedSegIdx)
    selectedSeg = seg3D[bs_view, selectedSegIdx.long(), :]  # (1,256,16,3)

    AA = selectedSeg - fpsPoint.unsqueeze(2).unsqueeze(2)
    AA = (AA ** 2).sum(-1)  # 1,256,16,2
    BB = torch.argmin(AA, dim=-1)[0]  # 256,16,1
    Ind1 = torch.arange(sn).view(sn, 1).repeat((1, gn))
    Ind2 = torch.arange(gn).view(1, gn).repeat((sn, 1))
    selectedSegLabel = selectedSegLabel[Ind1, Ind2, BB]

    Ind1 = torch.arange(sn).view(sn, 1, 1).repeat((1, gn, 2))
    Ind2 = torch.arange(gn).view(1, gn, 1).repeat((sn, 1, 2))
    CC = torch.argsort(AA, dim=-1)[0]
    selectedSeg = selectedSeg[:, Ind1, Ind2, CC, :]
    Mylabel = []
    for i in range(selectedSegLabel.shape[0]):
        if ballMask[i] == 0:
            Mylabel.append(-1)
        else:
            a, b = torch.unique(selectedSegLabel[i], dim=-1, return_counts=True)
            if a[0] == -1 and b[0] >= NegSampleThresh:
                labelA = a[0]
            elif a[0] == -1:
                a, b = a[1:], b[1:]
                argb = torch.argmax(b)
                labelA = a[argb]
            else:
                argb = torch.argmax(b)
                labelA = a[argb]
            Mylabel.append(labelA)

    Mylabel = torch.Tensor(Mylabel)
    classifyLabel = torch.where(Mylabel >= 0, 1, 0)
    X = selectedSeg

    X = X - fpsPoint.unsqueeze(2).unsqueeze(2)
    X = X.flatten(start_dim=-2, end_dim=-1)
    X = torch.cat([X, toCat], dim=-1)
    return X, Mylabel, classifyLabel, objGTJunc3D_array, fpsPoint, mean, max, name, np.array(
        objLineIdx), rec_label, word_mask, fpsLabel

