import numpy as np
import cv2 as cv
import json

def read_npz_file():
    '''
    ---This is the explanation of *.npz file in multi-view images dataset：---

    K:  3x3 camera intrinsics
    P: 3x4  P = K @ Rt
    Rt: 3x4;  3x3: rotation matrix, 3x1: transform matrix
    junc2D: N x 2 , 2D junctions in images
    junc3D_cam: N x 3, 3D junctions in camera coord system
    junc3D_world: N x 3, 3D junctions in world coord system
    edge: Nx3, (idx1,idx2,if_vis) , idx1/idx2: index of junction, if_vis(bool): if this edge is visible or not
    edgeWireframeIdx: edge corresponding to the gt wireframe edge index 
    '''
    npzFile = "E:\dataset\CG10_500_048067_0046\CG10_500_048067_0046_001.npz"
    data = np.load(npzFile)
    imgFile = "E:\dataset\CG10_500_048067_0046\CG10_500_048067_0046_001.png"
    img = cv.imread(imgFile)


    #visualize 2D lines in img ##
    junc2d = data['junc2D']
    edge = data['edge']

    for i in range(len(edge)):
        p1, p2, vis = edge[i]
        x1, x2 = junc2d[p1], junc2d[p2]
        x1 = [round(x1[0]), round(x1[1])]
        x2 = [round(x2[0]), round(x2[1])]
        if vis:
            cv.line(img, x1, x2, (255,0,0), 3, cv.LINE_AA)
        else:
            cv.line(img, x1, x2, (0,255,0), 2, cv.LINE_AA)
    cv.imshow("img with 2d lines",img)
    cv.waitKey(0)

    for k in data.keys():
        print(k,data[k].shape)


def read_json_file():
    # ---This is the explanation of *.json file in line cloud dataset：---
    '''
    junc3DList: N x 3, 3D junctions in world coord system
    edgeList: N x 2 (idx1, idx2), edge of the line cloud, idx1/idx2: index of the junc3DList

    junc2DList: N x 2, 2D junctions in a certain image/view.
    viewList: N x 1, 2D junction No.i corresponds to the image/view viewList[No.i], viewList is 1-based index. This parameter is come from Line3Dpp

    label: N x 1, junction No.i corresponds to the junction No.label[i] of the ground-truth wireframe junction. -1 means this junction belongs to noise.
    objGTJunc3D: N x 3, ground-truth junction of the 3D wireframe.
    objGTSeg3D: N x 6, ground-truth segment of the 3D wireframe. 6 means xyz of two points
    line_idx: ground-truth edge of the 3D wireframe.

    '''
    json_file = "E:/dataset/LineCloud_0130_P123/BJ39_500_098050_0009.json"
    data = json.load(open(json_file, 'r'))
    for k in data.keys():
        print(k, np.array(data[k]).shape)

