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


def ClassifyLineStaticAndDynamic(self, item):

    json_path = self.obj_list[item]
    pred_path = self.predJunc_list[item]
    name = os.path.split(json_path)[-1][:-5]
    data = json.load(open(json_path, 'r'))
    predData = json.load(open(pred_path, 'r'))

    # data from json
    junc3DList = data['junc3DList']

    # data from wireframe GT
    objGTJunc3D = data['objGTJunc3D']
    objLineIdx = data['line_idx']
    # 邻接矩阵
    Adjacency_matrix = np.zeros((len(objGTJunc3D), len(objGTJunc3D)))
    for i, j in objLineIdx:
        Adjacency_matrix[i, j] = Adjacency_matrix[j, i] = 1
    A0 = np.eye(len(objGTJunc3D) + 1)
    A0[-1, -1] = 0
    N = Adjacency_matrix.shape[0]
    Adjacency_matrix = np.concatenate(
        [np.concatenate([Adjacency_matrix, np.zeros((N, 1))], axis=-1), np.zeros((1, N + 1))], axis=0)

    A2 = Adjacency_matrix @ Adjacency_matrix

    A2 = (~Adjacency_matrix.astype(np.bool_)) & A2.astype(np.bool_) & (~A0.astype(np.bool_))  # ~A1 乘 A2 过滤掉1阶段临接点
    An = ~(Adjacency_matrix.astype(np.bool_) | A2.astype(np.bool_) | A0.astype(np.bool_))

    An_hardNeg = np.copy(An)
    An_hardNeg[:, -1] = 0
    An_hardNeg[-1, :] = 0

    An_easyNeg = np.copy(An)
    An_easyNeg[:-1, :-1] = 0

    # data from PredJunc
    # mean,max = torch.Tensor(predData['mean']), torch.tensor(predData['std'])
    mean, std = torch.Tensor(self.mean).float(), torch.tensor(self.std)
    predXYZ = torch.Tensor(predData['predXYZ']).float()
    # predXYZconfi = torch.softmax(torch.Tensor(predData['logits']).float(),dim=1)[:,1,:]
    logits = torch.Tensor(predData['logits']).float()
    confiAll = F.softmax(logits, dim=1)[:, 1, :]
    fpsPoint = torch.Tensor(predData['fpsPoint']).float()

    # inv-normalize
    predXYZ = (predXYZ + fpsPoint) * std + mean

    S1 = objLineIdx  #
    S2 = np.array((np.triu(A2, 1) > 0).nonzero()).transpose(1, 0).tolist()
    Sn = np.array((np.triu(An_hardNeg, 1) > 0).nonzero()).transpose(1, 0).tolist()
    S0 = []
    S = S0 + S1 + S2 + Sn
    staticClassifyLabel = len(S0) * [0] + len(S1) * [1] + len(S2) * [2] + len(Sn) * [3]
    sampleNum = 256  # 1/4的static
    if len(S) >= sampleNum:
        S = S[:sampleNum]
        staticClassifyLabel = staticClassifyLabel[:sampleNum]
    else:

        # S = (sampleNum//len(S)+1) * S
        S = S[:len(S)]
        # staticClassifyLabel = (sampleNum//len(staticClassifyLabel)+1) * staticClassifyLabel
        staticClassifyLabel = staticClassifyLabel[:len(S)]


    dynamicSampleNum = 1024 - len(S)
    bs = predXYZ.shape[0]
    for i in range(bs):
        pred, confi = predXYZ[i].tolist(), confiAll[i].tolist()
        pred, confi = self.postprocess.nms(pred, confi)
        pred, confi = pred[:100], confi[:100]
        pred, label, confi = self.postprocess.assignLabel(pred, objGTJunc3D, confi)
        label = label.tolist()
        # step1 vis
        # checkPointV2(pred.tolist(),label,objGTJunc3D,objLineIdx,name)

        DD = list(combinations(label, 2))
        Idx = list(range(len(pred)))
        EE = np.array(list(combinations(Idx, 2)))
        a, b = zip(*DD)


        D1_Ind = Adjacency_matrix[a, b].nonzero()[0]
        D1 = EE[D1_Ind, :].tolist()
        #
        D2_Ind = A2[a, b].nonzero()[0]
        D2 = EE[D2_Ind, :].tolist()
        #

        DnHard_Ind = An_hardNeg[a, b].nonzero()[0]
        DnHard = EE[DnHard_Ind, :].tolist()

        DnEasy_Ind = An_easyNeg[a, b].nonzero()[0]
        DnEasy = EE[DnEasy_Ind, :].tolist()

        #
        D0_Ind = A0[a, b].nonzero()[0]
        D0 = EE[D0_Ind, :].tolist()

        D = D0 + D1 + D2 + DnHard + DnEasy
        dynamicClassifyLabel = len(D0) * [0] + len(D1) * [1] + len(D2) * [2] + len(DnHard) * [3] + len(DnEasy) * [4]



        if len(D) > dynamicSampleNum:
            D = D[:dynamicSampleNum]
            dynamicClassifyLabel = dynamicClassifyLabel[:dynamicSampleNum]
        else:
            D = (dynamicSampleNum // len(D) + 1) * D
            D = D[:dynamicSampleNum]
            dynamicClassifyLabel = (dynamicSampleNum // len(dynamicClassifyLabel) + 1) * dynamicClassifyLabel
            dynamicClassifyLabel = dynamicClassifyLabel[:dynamicSampleNum]



        # step2 vis connection
        # visDynamicStep2(pred.tolist(), label, objGTJunc3D, objLineIdx, D, dynamicClassifyLabel, name)
        # D = list(combinations(Idx,2))
        # dynamicClassifyLabel = len(D) * [0]
    predXYZ = pred.unsqueeze(0)

    # normalize
    junc3D_array = np.array(junc3DList, dtype=np.float64)
    mean = np.array(self.mean, dtype=np.float64)
    junc3D_array = junc3D_array - mean[None, :]
    std = self.std
    junc3D_array = junc3D_array / std
    objGTJunc3D_array = np.array(objGTJunc3D, dtype=np.float64)
    objGTJunc3D_array = (objGTJunc3D_array - mean[None, :]) / std
    predXYZ_array = predXYZ.numpy()
    predXYZ_array = (predXYZ_array - mean[None, :]) / std

    # scale and shift
    if self.split == "train":
        junc3D_array, objGTJunc3D_array, predXYZ_array = dataAug(junc3D_array, objGTJunc3D_array, predXYZ_array)

    junc3D = torch.from_numpy(junc3D_array).float().unsqueeze(0)
    seg3D = junc3D.view(junc3D.shape[0], -1, 2, 3)  # (B,N,2,3)
    predXYZ = torch.from_numpy(predXYZ_array).float()  # (1,256,3) have normalized
    EachGroupNumber = 32



    # get dynamic data
    dynamicFpsPoint = predXYZ[:, D, :].view(-1, 3).unsqueeze(0)  # (bs,dynamicNum*2,3)
    selectedSegIdx, ballMask, _, dynamic_word_mask = query_ball_pointV3(0.2, EachGroupNumber, seg3D,
                                                                        dynamicFpsPoint)  # segIdx: (1,256,16)
    bs_view = torch.zeros_like(selectedSegIdx)
    selectedSeg = seg3D[bs_view, selectedSegIdx.long(), :]  # (1,256,16,2,3)  # 2,3代表一条线
    dynamicSelectedSeg = selectedSeg.view(1, -1, EachGroupNumber * 2, 2, 3)

    # get static data
    staticFpsPoint = torch.from_numpy(objGTJunc3D_array).float()[S, :].view(-1, 3).unsqueeze(
        0)  # (1,256,3) (bs,sampleNum*2,3)
    selectedSegIdx, ballMask, _, static_word_mask = query_ball_pointV3(0.2, EachGroupNumber, seg3D,
                                                                       staticFpsPoint)  # segIdx: (1,256,16)  # sample 16 segments based on fpsPoint, here return index
    bs_view = torch.zeros_like(selectedSegIdx)
    selectedSeg = seg3D[bs_view, selectedSegIdx.long(), :]  # (1,256,16,2,3)  # 2,3代表一条线
    staticSelectedSeg = selectedSeg.view(1, -1, EachGroupNumber * 2, 2, 3)

    # step3 vis possible connection
    # visDynamicStep3(dynamicSelectedSeg[0].tolist(), objGTJunc3D_array.tolist(), objLineIdx, D, dynamicClassifyLabel, dynamicFpsPoint[0].tolist(), name)
    # visDynamicStep3(staticSelectedSeg[0].tolist(), objGTJunc3D_array.tolist(), objLineIdx, S, staticClassifyLabel, staticFpsPoint[0].tolist(), name)

    fpsPoint = torch.cat([staticFpsPoint, dynamicFpsPoint], dim=1)  # (1,2048,3)
    fpsPoint = fpsPoint.view(1, -1, 2, 3)
    fpsPoint = (fpsPoint[:, :, 0, :] + fpsPoint[:, :, 1, :]) / 2
    selectedSeg = torch.cat([staticSelectedSeg, dynamicSelectedSeg], dim=1)
    word_mask = torch.cat([static_word_mask, dynamic_word_mask], dim=1)
    combine = S + D

    # endpoint permutation invariance
    _, sn, gn, _, _ = selectedSeg.shape
    AA = selectedSeg - fpsPoint.unsqueeze(2).unsqueeze(2)
    AA = (AA ** 2).sum(-1)  # 1,256,16,2
    Ind1 = torch.arange(sn).view(sn, 1, 1).repeat((1, gn, 2))
    Ind2 = torch.arange(gn).view(1, gn, 1).repeat((sn, 1, 2))
    CC = torch.argsort(AA, dim=-1)[0]
    selectedSeg = selectedSeg[:, Ind1, Ind2, CC, :]
    X = selectedSeg

    X = X - fpsPoint.unsqueeze(2).unsqueeze(2)
    classifyLabel = staticClassifyLabel + dynamicClassifyLabel
    X = X.flatten(start_dim=-2, end_dim=-1)


    if self.split == "train":
        return X, np.array(label), torch.Tensor(classifyLabel), objGTJunc3D_array, fpsPoint, mean, std, name, np.array(
            combine), confi.numpy(), predXYZ.numpy(), np.array(objLineIdx), word_mask, mean
    else:
        return X, np.array(label), torch.Tensor(classifyLabel), objGTJunc3D_array, fpsPoint, mean, std, name, np.array(
            combine), confi.numpy(), predXYZ.numpy(), np.array(objLineIdx), word_mask, mean


def ClassifyLineTestDynamicForTest(self, item):

    json_path = self.obj_list[item]
    pred_path = self.predJunc_list[item]
    name = os.path.split(json_path)[-1][:-5]
    data = json.load(open(json_path, 'r'))
    predData = json.load(open(pred_path, 'r'))

    # data from json
    junc3DList = data['junc3DList']

    # data from wireframe GT
    objGTJunc3D = data['objGTJunc3D']
    objLineIdx = data['line_idx']
    # 邻接矩阵
    Adjacency_matrix = np.zeros((len(objGTJunc3D), len(objGTJunc3D)))
    for i, j in objLineIdx:
        Adjacency_matrix[i, j] = Adjacency_matrix[j, i] = 1
    A0 = np.eye(len(objGTJunc3D) + 1)
    A0[-1, -1] = 0
    N = Adjacency_matrix.shape[0]
    Adjacency_matrix = np.concatenate(
        [np.concatenate([Adjacency_matrix, np.zeros((N, 1))], axis=-1), np.zeros((1, N + 1))], axis=0)

    A2 = Adjacency_matrix @ Adjacency_matrix
    A2 = (~Adjacency_matrix.astype(np.bool_)) & A2.astype(np.bool_) & (~A0.astype(np.bool_))  # ~A1 乘 A2 过滤掉1阶段临接点
    An = ~(Adjacency_matrix.astype(np.bool_) | A2.astype(np.bool_) | A0.astype(np.bool_))

    An_hardNeg = np.copy(An)
    An_hardNeg[:, -1] = 0
    An_hardNeg[-1, :] = 0

    An_easyNeg = np.copy(An)
    An_easyNeg[:-1, :-1] = 0

    # data from PredJunc
    # mean,max = torch.Tensor(predData['mean']), torch.tensor(predData['std'])
    mean, max = torch.Tensor(self.mean).float(), torch.tensor(self.std)
    predXYZ = torch.Tensor(predData['predXYZ']).float()
    logits = torch.Tensor(predData['logits']).float()
    confiAll = F.softmax(logits, dim=1)[:, 1, :]
    fpsPoint = torch.Tensor(predData['fpsPoint']).float()

    # inv-normalize
    predXYZ = (predXYZ + fpsPoint) * max + mean


    bs = predXYZ.shape[0]
    for i in range(bs):
        pred, confi = predXYZ[i].tolist(), confiAll[i].tolist()
        pred, confi = self.postprocess.nms(pred, confi)

        pred, confi = self.postprocess.confiMask(pred, confi)
        pred, label, confi = self.postprocess.assignLabel(pred, objGTJunc3D, confi)
        label = label.tolist()
        # step1 vis pred 3d junc and gt
        # checkPointV2(pred.tolist(),label,objGTJunc3D,objLineIdx,name)

        DD = list(combinations(label, 2))
        Idx = list(range(len(pred)))
        EE = np.array(list(combinations(Idx, 2)))
        a, b = zip(*DD)


        D1_Ind = Adjacency_matrix[a, b].nonzero()[0]
        D1 = EE[D1_Ind, :].tolist()
        #
        D2_Ind = A2[a, b].nonzero()[0]
        D2 = EE[D2_Ind, :].tolist()
        #

        DnHard_Ind = An_hardNeg[a, b].nonzero()[0]
        DnHard = EE[DnHard_Ind, :].tolist()

        DnEasy_Ind = An_easyNeg[a, b].nonzero()[0]
        DnEasy = EE[DnEasy_Ind, :].tolist()

        #
        D0_Ind = A0[a, b].nonzero()[0]
        D0 = EE[D0_Ind, :].tolist()

        D = D0 + D1 + D2 + DnHard + DnEasy
        dynamicClassifyLabel = len(D0) * [0] + len(D1) * [1] + len(D2) * [2] + len(DnHard) * [3] + len(DnEasy) * [4]

        # step2 vis dynamic connection
        # visDynamicStep2(pred.tolist(), label, objGTJunc3D, objLineIdx, D, dynamicClassifyLabel, name)
        # D = list(combinations(Idx,2))
        # dynamicClassifyLabel = len(D) * [0]
    predXYZ = pred.unsqueeze(0)
    # normalize
    junc3D_array = np.array(junc3DList, dtype=np.float64)
    mean = np.array(self.mean, dtype=np.float64)

    junc3D_array = junc3D_array - mean[None, :]

    max = self.std
    junc3D_array = junc3D_array / max
    objGTJunc3D_array = np.array(objGTJunc3D, dtype=np.float64)
    objGTJunc3D_array = (objGTJunc3D_array - mean[None, :]) / max
    predXYZ_array = predXYZ.numpy()
    predXYZ_array = (predXYZ_array - mean[None, :]) / max

    # scale and shift
    if self.split == "train":
        junc3D_array, objGTJunc3D_array, predXYZ_array = dataAug(junc3D_array, objGTJunc3D_array, predXYZ_array)

    junc3D = torch.from_numpy(junc3D_array).float().unsqueeze(0)
    seg3D = junc3D.view(junc3D.shape[0], -1, 2, 3)  # (B,N,2,3)
    predXYZ = torch.from_numpy(predXYZ_array).float()  # (1,256,3) have normalized

    # get dynamic data
    EachGroupNumber = 32
    dynamicFpsPoint = predXYZ[:, D, :].view(-1, 3).unsqueeze(0)  # (bs,dynamicNum*2,3) # *2: line segments have two endepoints
    selectedSegIdx, ballMask, _, word_mask = query_ball_pointV3(0.2, EachGroupNumber, seg3D,
                                                                dynamicFpsPoint)  # segIdx: (1,256,16)  # sample 16 lines based on fpspoint, return index here
    bs_view = torch.zeros_like(selectedSegIdx)
    selectedSeg = seg3D[bs_view, selectedSegIdx.long(), :]  # (1,256,16,2,3)  # 2,3代表一条线
    dynamicSelectedSeg = selectedSeg.view(1, -1, EachGroupNumber * 2, 2, 3)

    # step3 vis possible connection
    # visDynamicStep3(dynamicSelectedSeg[0].tolist(), objGTJunc3D_array.tolist(), objLineIdx, D, dynamicClassifyLabel, dynamicFpsPoint[0].tolist(), name)
    # visDynamicStep3(staticSelectedSeg[0].tolist(), objGTJunc3D_array.tolist(), objLineIdx, S, staticClassifyLabel, staticFpsPoint[0].tolist(), name)
    # visDynamicStep3_Figure2(dynamicSelectedSeg[0].tolist(), objGTJunc3D_array.tolist(), objLineIdx, D, dynamicClassifyLabel, dynamicFpsPoint[0].tolist(), name, torch.Tensor(self.mean),torch.tensor(self.std),DD)

    fpsPoint = dynamicFpsPoint
    fpsPoint = fpsPoint.view(1, -1, 2, 3)
    fpsPoint = (fpsPoint[:, :, 0, :] + fpsPoint[:, :, 1, :]) / 2
    selectedSeg = dynamicSelectedSeg
    combine = D
    edgeMask = [0] * len(D)
    # endpoint permutation invariance

    _, sn, gn, _, _ = selectedSeg.shape
    AA = selectedSeg - fpsPoint.unsqueeze(2).unsqueeze(2)
    AA = (AA ** 2).sum(-1)  # 1,256,16,2
    Ind1 = torch.arange(sn).view(sn, 1, 1).repeat((1, gn, 2))
    Ind2 = torch.arange(gn).view(1, gn, 1).repeat((sn, 1, 2))
    CC = torch.argsort(AA, dim=-1)[0]
    selectedSeg = selectedSeg[:, Ind1, Ind2, CC, :]
    X = selectedSeg

    X = X - fpsPoint.unsqueeze(2).unsqueeze(2)
    classifyLabel = dynamicClassifyLabel
    X = X.flatten(start_dim=-2, end_dim=-1)

    return X, np.array(label), torch.Tensor(classifyLabel), objGTJunc3D_array, fpsPoint, mean, max, name, np.array(
        combine), confi.numpy(), predXYZ.numpy(), np.array(objLineIdx), word_mask, edgeMask



