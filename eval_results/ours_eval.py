import sys
# print(sys.path)
# sys.path.append("/data/code/LC2WF/eval_results")
# sys.path.append("./")
from graph import Graph
# import open3d
from misc.colors import colormap_255, semantics_cmap
import numpy as np
import os
import json
import glob
from itertools import permutations
import copy
from metric_wed import WED
import json
from functools import cmp_to_key
from metric import edgeSap,getsap
import time
import shutil
'''

eval ours result
save some results
open3d visualize
'''
def cmp(x,y):
    if x[1]<y[1]:
        return -1
    elif x[1]>y[1]:
        return 1
    else:
        if x[2]<y[2]:
            return -1
        elif x[2]>y[2]:
            return 1
        else:
            return 0

def cmp_filesize(x,y):
    if x[-1]<y[-1]:
        return -1
    elif x[-1]>y[-1]:
        return 1
    else:
        return 0
def cmp_gtjunc(x,y):
    if x[-2]<y[-2]:
        return -1
    elif x[-2]>y[-2]:
        return 1
    else:
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


def eval_metric(name,predXYZ,predConnect,wireframeJunc,wireframeLine,mean,std):
    OBJGT_PATH = "/data/obj_data/obj_vt/house"
    xx,_,yy = readWireframeGT(os.path.join(OBJGT_PATH,name.replace('.json','.obj')))
    mean,std = np.array(mean,dtype=np.float64),np.array(std,dtype=np.float64)
    # wireframeJunc = np.array(wireframeJunc,dtype=np.float64)
    # wireframeJunc = (wireframeJunc * std) + mean[None,:]
    wed = WED(wireframeJunc=wireframeJunc,wireframeLine=wireframeLine,predJunc=predXYZ,predEdge=predConnect)
    return wed.match_junc()


def p_info(data,info):
    data = np.array(data)
    print(info,"min {} max {} avg {} std {}".format(np.min(data),np.max(data),np.mean(data),np.std(data)))


def dynamicMatchV2():
    start_time  = time.time()
    all_time = 0
    FinalOutPath = "./finalOutOBJ"
    os.makedirs(FinalOutPath,exist_ok=True)
    json_list = sorted(glob.glob("./log/outputPredWireframe/*.json"))
    LINE_NMS_THRESH = 20
    CONNECT_THRESH = 0.30
    MERGE_THRESH = 0.2


    # direct output
    Jap3 = getsap(s=3,nms_threshhold=1,confi_thresh=0)
    Jap5 = getsap(s=5,nms_threshhold=1,confi_thresh=0)
    Jap7 = getsap(s=7,nms_threshhold=1,confi_thresh=0)

    Jap3_nms = getsap(s=3,nms_threshhold=1,confi_thresh=0)
    Jap5_nms = getsap(s=5,nms_threshhold=1,confi_thresh=0)
    Jap7_nms = getsap(s=7,nms_threshhold=1,confi_thresh=0)

    Jap3_nms_post = getsap(s=3,nms_threshhold=1,confi_thresh=0)
    Jap5_nms_post = getsap(s=5,nms_threshhold=1,confi_thresh=0)
    Jap7_nms_post = getsap(s=7,nms_threshhold=1,confi_thresh=0)

    Eap10 = edgeSap(s=10,nms_threshhold=0.01,confi_thresh=0)
    Eap7 = edgeSap(s=7, nms_threshhold=0.01,confi_thresh=0)
    Eap5 = edgeSap(s=5,nms_threshhold=0.01,confi_thresh=0)

    Eap10_nms = edgeSap(s=10,nms_threshhold=LINE_NMS_THRESH,confi_thresh=0)
    Eap7_nms = edgeSap(s=7, nms_threshhold=LINE_NMS_THRESH,confi_thresh=0)
    Eap5_nms = edgeSap(s=5,nms_threshhold=LINE_NMS_THRESH,confi_thresh=0)


    Eap10_nms_post = edgeSap(s=10,nms_threshhold=0,confi_thresh=0)
    Eap7_nms_post = edgeSap(s=7, nms_threshhold=0,confi_thresh=0)
    Eap5_nms_post = edgeSap(s=5,nms_threshhold=0,confi_thresh=0)


    eval_wed = []  # before nms
    eval_wed_nms = []  # after nms
    eval_wed_postprocess = [] # after postprocess
    

    output_vertex_size = []
    output_edge_size = []
    output_vertex_size_before_process = []
    output_edge_size_before_process = []


    for ii,js in enumerate(json_list):

        name = os.path.split(js)[-1]
        VIS_LIST = []
        f = open(js,'r')

        ### read data ###
        data = json.load(f)
        predXYZ = np.array(data['predXYZ'])[0]
        predXYZconfi = np.array(data['predXYZconfi'])
        prob = np.array(data['prob'])
        combine = np.array(data['combine'])
        label  = data['label']
        wireframeJunc = np.array(data['wireframeJunc'])
        wireframeLine = data['wireframeLine']
        mean,std = data['mean'], data['std']
        color = [tuple(np.random.randint(0,255,3,dtype=int).tolist()) for _ in range(wireframeJunc.shape[0])] + [(0,0,0)]
        # inv-normalize

        predXYZ = (predXYZ * np.array(std))  + np.array(mean)[None,:]
        wireframeJunc = (wireframeJunc*np.array(std)) + np.array(mean)[None,:]

        Adj_connect = np.zeros((predXYZ.shape[0],predXYZ.shape[0]))
        Adj_connect[combine[:,0],combine[:,1]] = Adj_connect[combine[:,1],combine[:,0]] = prob[1]


        merge,connect,_,disconnect,lost = prob[0],prob[1],prob[2],prob[3],prob[4]

        Connect_thresh = CONNECT_THRESH
        Merge_Thresh = MERGE_THRESH
        selectIdx = (merge>Merge_Thresh).nonzero()[0]  #  classify to connected
        if len(selectIdx)==0:
            print("find no edges in {}:{}".format(str(ii),js))
            print("min/max merge:{:.2f},{:.2f}".format(merge.min(),merge.max()))
        else:
            print("process {}:{}".format(str(ii), js))
        selectCombine = combine[selectIdx]
        

        #direct output
        BEFORE_PROCESS_PATH = "./before_process"
        os.makedirs(BEFORE_PROCESS_PATH,exist_ok=True)
        sI  = (connect>Connect_thresh).nonzero()[0]
        sC = combine[sI].tolist() #edge conbine
        sW = connect[sI] # edge weight

        output_vertex_size_before_process.append(predXYZ.shape[0])
        output_edge_size_before_process.append(sC.__len__())

        Jap3(predXYZ,predXYZconfi,wireframeJunc)  # predXYZ and wireframe junc inv-normalize
        Jap5(predXYZ,predXYZconfi,wireframeJunc)
        Jap7(predXYZ,predXYZconfi,wireframeJunc)

        # test why sap get worse after post-process
        e5 = edgeSap(s=5,nms_threshhold=0,confi_thresh=0)
        e5(predXYZ,sC,sW,wireframeJunc,wireframeLine)
        tempG = Graph(predXYZ,sC,sW,predXYZconfi,line_nms_thresh=LINE_NMS_THRESH)
        VIS_LIST.append(tempG.ret_clean_data())

        Eap5(predXYZ,sC,sW,wireframeJunc,wireframeLine)
        Eap7(predXYZ,sC,sW,wireframeJunc,wireframeLine)
        Eap10(predXYZ,sC,sW,wireframeJunc,wireframeLine)
        # wed before nms
        eval_wed.append(eval_metric(name,predXYZ,sC,wireframeJunc.tolist(),wireframeLine,mean,std))

        # results after line nms
        # save obj
        g_before_nms = Graph(predXYZ,sC,sW,predXYZconfi,line_nms_thresh=LINE_NMS_THRESH)
        g_before_nms.save_clean_obj(os.path.join(BEFORE_PROCESS_PATH,name.replace(".json",".obj")))

        g_before_nms.lineNms()
        vertex_before_nms,edge_before_nms = g_before_nms.ret_clean_data()
        eval_wed_nms.append(eval_metric(name,vertex_before_nms,edge_before_nms,wireframeJunc,wireframeLine,mean,std))
        new_edge = g_before_nms.retPredEdge()
        #for eval junc ap
        new_vert = predXYZ[np.unique(np.array(new_edge).reshape(-1)),:]
        new_confi = predXYZconfi[np.unique(np.array(new_edge).reshape(-1))]
        Jap3_nms(new_vert,new_confi,wireframeJunc)
        Jap5_nms(new_vert,new_confi,wireframeJunc)
        Jap7_nms(new_vert,new_confi,wireframeJunc)


        Eap5_nms(predXYZ,new_edge,g_before_nms.edgeWeight,wireframeJunc,wireframeLine)
        Eap7_nms(predXYZ,new_edge,g_before_nms.edgeWeight,wireframeJunc,wireframeLine)
        Eap10_nms(predXYZ,new_edge,g_before_nms.edgeWeight,wireframeJunc,wireframeLine)
        eval_wed_nms.append(eval_metric(name,*g_before_nms.ret_clean_data(),wireframeJunc.tolist(), wireframeLine,mean,std))


        t1 = time.time()
        FF = list(zip(selectIdx.tolist(),merge[selectIdx].tolist()))
        FF = sorted(FF,key=lambda x:x[1],reverse=True)


        selectIdx, _ = zip(*FF)
        selectCombine = combine[np.array(selectIdx)]
        junc_to_delete=[]
        for i in range(selectCombine.shape[0]):
            a,b = selectCombine[i][0],selectCombine[i][1]
            if predXYZconfi[a]<predXYZconfi[b]: a,b = b,a
            if b not in junc_to_delete:
                junc_to_delete.append(b)
                Adj_connect[a] = np.maximum(Adj_connect[a],Adj_connect[b])
        new_merge = Adj_connect[combine[:,0],combine[:,1]]
        merge = new_merge

        selectIdx = (connect>Connect_thresh).nonzero()[0]
        selectWeight = connect[selectIdx]
        selectCombine = combine[selectIdx]
        selectCombine = selectCombine.tolist()
        selectCombine = [i for i in selectCombine if i[0] not in junc_to_delete and i[1] not in junc_to_delete]


        g = Graph(predXYZ,selectCombine,selectWeight,predXYZconfi,line_nms_thresh=LINE_NMS_THRESH)
        VIS_LIST.append(g.ret_clean_data())
        g.removeDegreeOne()


        VIS_LIST.append(g.ret_clean_data())
        
        g.juncNMS()
        VIS_LIST.append(g.ret_clean_data())
        g.lineNms()
        VIS_LIST.append(g.ret_clean_data())
        # g.juncNMS()

        # g.removeDegreeOne()
        g.removeDegreeOneV2() # remove edge both side degree=1
        all_time = all_time + time.time()-t1
        # g.cal_dis()

        predConnect = g.retPredEdge()
        # eval post-process start
        # only eval junc ap
        predXYZ_post = predXYZ[np.unique(np.array(predConnect).reshape(-1)),:]
        predXYZconfi_post = predXYZconfi[np.unique(np.array(predConnect).reshape(-1))]
        Jap3_nms_post(predXYZ_post,predXYZconfi_post,wireframeJunc)
        Jap5_nms_post(predXYZ_post,predXYZconfi_post,wireframeJunc)
        Jap7_nms_post(predXYZ_post,predXYZconfi_post,wireframeJunc)

        
        # name = os.path.split(js)[-1].replace(".json","")
        g.save_clean_obj(os.path.join(FinalOutPath,name.replace(".json",".obj")))
        predXYZ,predConnect = g.ret_clean_data()
        
        # test sap get worse after post-process
        e5_post = edgeSap(s=5,nms_threshhold=0,confi_thresh=0)
        e5_post(predXYZ,predConnect,g.edgeWeight,wireframeJunc,wireframeLine)

        Eap5_nms_post(predXYZ,predConnect,g.edgeWeight,wireframeJunc,wireframeLine,)
        Eap7_nms_post(predXYZ,predConnect,g.edgeWeight,wireframeJunc,wireframeLine,)
        Eap10_nms_post(predXYZ,predConnect,g.edgeWeight,wireframeJunc,wireframeLine)

        output_vertex_size.append(len(predXYZ))
        output_edge_size.append(len(predConnect))
        eval_wed_postprocess.append(eval_metric(name,predXYZ,predConnect,wireframeJunc.tolist(),wireframeLine,mean,std))


    Jap3_result = Jap3.get_RPF()
    Jap5_result = Jap5.get_RPF()
    Jap7_result = Jap7.get_RPF()

    Eap5_result = Eap5.get_RPF()
    Eap7_result = Eap7.get_RPF()
    Eap10_result = Eap10.get_RPF()

    Jap3_nms_result = Jap3_nms.get_RPF()
    Jap5_nms_result = Jap5_nms.get_RPF()
    Jap7_nms_result = Jap7_nms.get_RPF()

    Eap5_nms_result = Eap5_nms.get_RPF()
    Eap7_nms_result = Eap7_nms.get_RPF()
    Eap10_nms_result = Eap10_nms.get_RPF()

    Jap3_nms_post_result = Jap3_nms_post.get_RPF()
    Jap5_nms_post_result = Jap5_nms_post.get_RPF()
    Jap7_nms_post_result = Jap7_nms_post.get_RPF()

    Eap5_nms_post_result = Eap5_nms_post.get_RPF()
    Eap7_nms_post_result = Eap7_nms_post.get_RPF()
    Eap10_nms_post_result = Eap10_nms_post.get_RPF()


    wed_result = np.array(eval_wed).mean(axis=0)
    wed_nms_result = np.array(eval_wed_nms).mean(axis=0)
    wed_nms_post_result = np.array(eval_wed_postprocess).mean(axis=0)
    print("""
    #ours:
    before post-process, junc confi thresh 0.8, connect confi {}
    vtx {:.4f} {:.4f} {:.4f} {:.4f}
    line {:.4f} {:.4f} {:.4f} {:.4f}
    junc AP:
    ap recall 3 5 7 and mean: {:.4f} {:.4f}  {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}

    sAP
    Sap,recall 5 7 10 and mean: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}
    ##Wed
    # dist/20
    gt wireframe junc/line {} {}
    pred junc/line {} {}
    +junc and dist || +edge and dist || -edge and dist ||   total      ||  edge total
     {:.4f} {:.4f}    {:.4f} {:.4f}     {:.4f} {:.4f}    {:.4f} {:.4f}  || {:.4f} {:.4f}


    after post-process
    vtx {:.4f} {:.4f} {:.4f} {:.4f}
    line {:.4f} {:.4f} {:.4f} {:.4f}
    junc AP:
    ap recall 3 5 7 and mean: {:.4f} {:.4f}  {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}
    attention ## precision {:.4f} {:.4f} {:.4f}

    sAP
    Sap,recall 5 7 10 and mean: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}
    ##Wed
    # dist/20
    gt wireframe junc/line {} {}
    pred junc/line {} {}
    +junc and dist || +edge and dist || -edge and dist ||   total      ||  edge total
     {:.4f} {:.4f}    {:.4f} {:.4f}     {:.4f} {:.4f}    {:.4f} {:.4f}  || {:.4f} {:.4f}

    """.format(CONNECT_THRESH,*Jap3_result['pred_stat'],*Eap5_result['pred_stat'],
    Jap3_result['AP'],Jap3_result['recall'],Jap5_result['AP'],Jap5_result['recall'],Jap7_result['AP'],Jap7_result['recall'],
    (Jap3_result['AP']+Jap5_result['AP']+Jap7_result['AP'])/3,
    (Jap3_result['recall']+Jap5_result['recall']+Jap7_result['recall'])/3,

    Eap5_result['AP'],Eap5_result['recall'],Eap7_result['AP'],Eap7_result['recall'],Eap10_result['AP'],Eap10_result['recall'],
    (Eap5_result['AP']+Eap7_result['AP']+Eap10_result['AP'])/3,
    (Eap5_result['recall']+Eap7_result['recall']+Eap10_result['recall'])/3,

    wed_result[0],wed_result[1],wed_result[2],wed_result[3],
    wed_result[5], wed_result[4]/20,
    wed_result[6], wed_result[9]/20,
    wed_result[7], wed_result[10]/20,
    wed_result[8]+wed_result[5], wed_result[11]/20 + wed_result[4]/20,
    wed_result[8], wed_result[11]/20 ,    

    *Jap3_nms_post_result['pred_stat'],*Eap5_nms_post_result['pred_stat'],
    Jap3_nms_post_result['AP'],Jap3_nms_post_result['recall'],Jap5_nms_post_result['AP'],Jap5_nms_post_result['recall'],Jap7_nms_post_result['AP'],Jap7_nms_post_result['recall'],
    (Jap3_nms_post_result['AP']+Jap5_nms_post_result['AP']+Jap7_nms_post_result['AP'])/3,
    (Jap3_nms_post_result['recall']+Jap5_nms_post_result['recall']+Jap7_nms_post_result['recall'])/3,
    Jap3_nms_post_result['precision'],Jap5_nms_post_result['precision'],Jap7_nms_post_result['precision'],
    Eap5_nms_post_result['AP'],Eap5_nms_post_result['recall'],Eap7_nms_post_result['AP'],Eap7_nms_post_result['recall'],Eap10_nms_post_result['AP'],Eap10_nms_post_result['recall'],
    (Eap5_nms_post_result['AP']+Eap7_nms_post_result['AP']+Eap10_nms_post_result['AP'])/3,
    (Eap5_nms_post_result['recall']+Eap7_nms_post_result['recall']+Eap10_nms_post_result['recall'])/3,

    wed_nms_post_result[0],wed_nms_post_result[1],wed_nms_post_result[2],wed_nms_post_result[3],
    wed_nms_post_result[5], wed_nms_post_result[4]/20,
    wed_nms_post_result[6], wed_nms_post_result[9]/20,
    wed_nms_post_result[7], wed_nms_post_result[10]/20,
    wed_nms_post_result[8]+wed_nms_post_result[5], wed_nms_post_result[11]/20 + wed_nms_post_result[4]/20,   
    wed_nms_post_result[8], wed_nms_post_result[11]/20,   
    ))


if __name__=="__main__":
    dynamicMatchV2()