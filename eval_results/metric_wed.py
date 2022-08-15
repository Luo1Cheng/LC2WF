import numpy as np
import os

'''
wed的度量
DATE: 2022-03-16

'''
class WED:
    def __init__(self, wireframeJunc, wireframeLine, predJunc, predEdge):
        self.WFJunc = wireframeJunc
        self.WFEdge = wireframeLine
        self.predJunc = predJunc
        self.predEdge = predEdge

        self.num_gt_junc = len(self.WFJunc)
        self.num_gt_edge = len(self.WFEdge)

        self.num_pred_junc = len(self.predJunc)
        self.num_pred_edge = len(self.predEdge)



    def line_to_line_dist(self, x, y):
        dis = ((x[:, None, :, None] - y[:, None]) ** 2).sum(-1)
        dis = np.sqrt(dis)
        dis = np.minimum(
            dis[:, :, 0, 0] + dis[:, :, 1, 1], dis[:, :, 0, 1] + dis[:, :, 1, 0]
        )
        return dis

    def match_junc(self):
        WFJunc = np.array(self.WFJunc, dtype=np.float64)
        predJunc = np.array(self.predJunc, dtype=np.float64)
        dist = np.sqrt(np.linalg.norm(predJunc[:, None, :] - WFJunc[None, :, :],axis=-1))  # eucliean dis
        N = WFJunc.shape[0]
        S = predJunc.shape[0]

        # 每个pred junc assigan到最近的 gt junc
        arg_dist = np.argmin(dist,axis=-1)
        edit_dis_junc = np.min(dist,axis=-1)

        #transform junc
        transform_junc = WFJunc[arg_dist]


        #add junc
        matched_gt_junc_index = np.unique(arg_dist)
        junc_add_num = WFJunc.shape[0] - matched_gt_junc_index.shape[0]

        #edge deletion
        predJunc_index = arg_dist # transform后对应gt的实际index
        #计算新的edge
        predEdge_new = arg_dist[np.array(self.predEdge)] #index是和gtwireframe的index对应的
        hitEdge = [0] * self.WFEdge.__len__()
        edge_to_delete = []
        for i,e in enumerate(predEdge_new.tolist()):
            if e in self.WFEdge or e[::-1] in self.WFEdge:
                ind = self.WFEdge.index(e) if e in self.WFEdge else self.WFEdge.index(e[::-1])
                if hitEdge[ind]==1:
                    edge_to_delete.append(i)
                else:
                    hitEdge[ind]=1
        # edge insertion
        edge_to_add = [i for i in range(len(hitEdge)) if hitEdge[i]==0]
        edge_add_num = edge_to_add.__len__()
        edge_delete_num = edge_to_delete.__len__()

        edge_to_add_data = np.array(self.WFJunc)[self.WFEdge,:][edge_to_add]
        edge_to_delete_data = transform_junc[np.array(self.predEdge)][edge_to_delete]
        edge_to_add_data = np.linalg.norm(edge_to_add_data[:,0] - edge_to_add_data[:,1],axis=-1)
        edge_to_delete_data = np.linalg.norm(edge_to_delete_data[:, 0] - edge_to_delete_data[:, 1], axis=-1)
        edge_edit_dis = edge_to_add_data.sum() + edge_to_delete_data.sum()


        print("""num of gt wireframe junc/line {},{},
        num of pred junc/line {},{},
        sum edit distance of junc {},
        add junc number {},
        edge need to be edit(sum of add and delete) {}+{}={},
        edge edit dis(mean length of add and delete edge) {}+{}={},
        """.format(self.WFJunc.__len__(),self.WFEdge.__len__(),self.predJunc.__len__(),self.predEdge.__len__(),edit_dis_junc.sum(),junc_add_num,
                   edge_add_num,edge_delete_num,edge_add_num+edge_delete_num,
                   edge_to_add_data.sum(),edge_to_delete_data.sum(),edge_edit_dis)
              )

        return [self.WFJunc.__len__(),self.WFEdge.__len__(),self.predJunc.__len__(),self.predEdge.__len__(),edit_dis_junc.sum(),junc_add_num,
                   edge_add_num,edge_delete_num,edge_add_num+edge_delete_num,
                   edge_to_add_data.sum(),edge_to_delete_data.sum(),edge_edit_dis]


    def match_edge(self):
        WFJunc = np.array(self.WFJunc, dtype=np.float64)
        predJunc = np.array(self.predJunc, dtype=np.float64)

        WFEdge_data = WFJunc[np.array(self.WFEdge), :]
        predEdge_data = predJunc[np.array(self.predEdge), :]

        N = WFEdge_data.shape[0]
        S = predEdge_data.shape[0]
        dist = self.line_to_line_dist(predEdge_data, WFEdge_data)

        choice = np.argmin(dist, axis=-1)
        dist = np.min(dist, axis=-1)[0]
        hit = np.zeros(N, dtype=np.bool)
        tp = np.zeros(dist.shape[0], dtype=np.float64)
        fp = np.zeros(dist.shape[0], dtype=np.float64)  # junc to delete
        tp_edit_dis = np.zeros(dist.shape[0], dtype=np.float64)
        for i in range(dist.shape[0]):
            if not hit[choice[i]]:
                hit[choice[i]] = True
                tp[i] = 1
                tp_edit_dis[i] = dist[i]
            else:
                fp[i] = 1
        edit_dis_all = tp_edit_dis.sum()
        matched_edge = tp.sum()
        edge_to_delele = fp.sum()
        edge_to_add = hit.shape[0] - hit.sum()

        self.num_match_edge = matched_edge
        self.num_delete_edge = edge_to_delele
        self.num_add_edge = edge_to_add
        self.edge_edit_dis = edit_dis_all

    def junc_result(self):
        print("""num of wireframe junc/line {}/{},
        num of pred junc/line {}/{}, 
        junc: match/delete/add/dis {}/{}/{}/{}, 
        edge: match/delete/add/dis {}/{}/{}/{},""".format(self.num_gt_junc, self.num_gt_edge, self.num_pred_junc,
                                                          self.num_pred_edge,
                                                          self.num_match_junc, self.num_delete_junc, self.num_add_junc,
                                                          self.junc_edit_dis,
                                                          self.num_match_edge, self.num_delete_edge, self.num_add_edge,
                                                          self.edge_edit_dis,
                                                          ))


