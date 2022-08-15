import numpy as np
'''
OURS 后处理主要使用的代码
DATE: 2022-03-16
'''

class Graph:
    def __init__(self,vert,edge,edgeWeight,confi,line_nms_thresh=20,):
        # junc confi has been sorted and must be sorted
        self.vert = vert
        self.vertConfi = confi
        
        ALL = list(zip(edge,edgeWeight))
        ALL = sorted(ALL,key=lambda x:x[1])
        edge,edgeWeight = zip(*ALL)
        

        self.edge = edge
        self.edgeWeight = edgeWeight
        self.edge_store = np.ones((len(self.edge)))

        self.updateAdjacency()

        self.line_nms_thresh = line_nms_thresh
        self.junc_hamming_thresh = 3
    def removeDegreeOne(self):
        # remove 两边度为1的边
        for i,(e1,e2) in enumerate(self.edge):
            if self.vert_degree[e1]<=1 and self.vert_degree[e2]<=1:
                self.edge_store[i]=0
        self.updateAdjacency()

    def removeDegreeOneV2(self):
        # remove 1边度为1的边
        for i,(e1,e2) in enumerate(self.edge):
            if self.vert_degree[e1]<=1 or self.vert_degree[e2]<=1:
                self.edge_store[i]=0
        self.updateAdjacency()
 
    def updateAdjacency(self):
        #根据边的store信息, 更新邻接矩阵，让邻接矩阵之储存没有删掉的边
        self.edge = [self.edge[i] for i,value in enumerate(self.edge_store) if value==1]
        self.edgeWeight = [self.edgeWeight[i] for i,value in enumerate(self.edge_store) if value==1]
        self.Adjacency = np.zeros((len(self.vert),len(self.vert)))
        for e1,e2 in self.edge:
            self.Adjacency[e1,e2]=1
            self.Adjacency[e2,e1]=1
        self.vert_degree = np.sum(self.Adjacency,axis=-1)
        self.edge_store = np.ones((len(self.edge))).tolist()

    def line_to_line_dist(self,x,y):
        dis = ((x[:,None,:, None] - y[:,None])**2).sum(-1)
        dis = np.sqrt(dis)
        dis = np.minimum(
            dis[:, :, 0, 0] + dis[:, :, 1, 1], dis[:, :, 0, 1] + dis[:, :, 1, 0]
        )
        return dis

    def lineNms(self):
        line = np.array(self.vert)[self.edge,:]
        dis = self.line_to_line_dist(line,line)
        dropped_line_index = []
        thresh=self.line_nms_thresh
        for i in range(dis.shape[0]):
            if i in dropped_line_index:
                continue
            d = dis[i]
            same_line_indexes = (d<thresh).nonzero()[0]
            for same_line_index_i in same_line_indexes:
                if same_line_index_i == i:
                    continue
                else:
                    dropped_line_index.append(same_line_index_i)
        for ii in dropped_line_index:
            self.edge_store[ii]=0
        self.updateAdjacency()


    def retPredEdge(self):
        ret = [self.edge[i] for i,value in enumerate(self.edge_store) if value==1]
        return ret

    
    def juncNMS(self):
        hamming = (self.Adjacency[:,None,:]!=self.Adjacency[None,:,:])
        dis = np.count_nonzero(hamming,axis=-1)
        thresh=self.junc_hamming_thresh
        dropped_junc_index=[]
        for i in range(dis.shape[0]):
            if i in dropped_junc_index:
                continue
            d = dis[i]
            same_line_indexes = (d<thresh).nonzero()[0]
            for same_line_index_i in same_line_indexes:
                if same_line_index_i == i:
                    continue
                else:
                    if np.linalg.norm(np.array(self.vert[i])-np.array(self.vert[same_line_index_i]))<30: #这是原始论文的值
                    # if np.linalg.norm(np.array(self.vert[i])-np.array(self.vert[same_line_index_i]))<20:
                        dropped_junc_index.append(same_line_index_i)
                        # dropped的时候也许需要把drop的连接关系merge一下？ 
                        # 发现merge反而会影响wed效果, 于是不再merge, 想得到原始的论文指标,需要merge
                        # xxx = self.Adjacency[same_line_index_i]
                        # yyy = xxx.nonzero()[0] #yyy和谁相连
                        # for kk in yyy:
                        #     if [i,kk] not in self.edge and [kk,i] not in self.edge:
                        #         self.edge.append([i,kk])
                        #         aaa  = self.edge.index([same_line_index_i,kk]) if [same_line_index_i,kk] in self.edge else self.edge.index([kk,same_line_index_i])
                        #         self.edgeWeight.append(self.edgeWeight[aaa])
                        #         self.edge_store.append(1)
        for i,(e1,e2) in enumerate(self.edge):
            if e1 in dropped_junc_index or e2 in dropped_junc_index:
                self.edge_store[i]=0
        self.updateAdjacency()

    def cal_dis(self):
        def line_to_line_distV1(x,y): #基于overlap merge一些edge
            pass
        vert = np.array(self.vert,dtype=np.float64)
        edge = self.retPredEdge()
        edge = np.array(edge)
        data = vert[edge,:]
    

    def save_clean_obj(self,outpath):
        predEdge = self.retPredEdge()
        vert = np.array(self.vert,dtype=np.float64)
        MD = {}
        new_vert = []
        num = 1
        for e1,e2 in predEdge:
            if e1 not in MD.keys():
                MD[e1] = num
                new_vert.append(vert[e1])
                num+=1

            if e2 not in MD.keys():
                MD[e2] = num
                new_vert.append(vert[e2])
                num+=1
        for ii,(e1,e2) in enumerate(predEdge):
            predEdge[ii]=[MD[e1],MD[e2]]
        with open(outpath,'w') as fw:
            for ii in range(new_vert.__len__()):
                fw.write("v "+" ".join([str(kk) for kk in new_vert[ii]])+"\n")
            for e1,e2 in predEdge:
                fw.write("l {} {}\n".format(e1,e2))
            
    def ret_clean_data(self):
        predEdge = self.retPredEdge()
        vert = np.array(self.vert,dtype=np.float64)
        MD = {}
        new_vert = []
        num = 0
        for e1,e2 in predEdge:
            if e1 not in MD.keys():
                MD[e1] = num
                new_vert.append(vert[e1])
                num+=1

            if e2 not in MD.keys():
                MD[e2] = num
                new_vert.append(vert[e2])
                num+=1
        for ii,(e1,e2) in enumerate(predEdge):
            predEdge[ii]=[MD[e1],MD[e2]]
        return [new_vert,predEdge]
        