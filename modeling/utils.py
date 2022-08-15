import torch
import numpy as np
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    if npoint>=N:
        x = np.arange(npoint)%N
        return torch.from_numpy(x).unsqueeze(0)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # farthest = torch.zeros((B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def farthest_point_sampleV2(xyz, npoint, oldidx):
    """
    有一些初始fps点的fps采样
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
        oldidx: [1,M]
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    if npoint>=N:
        x = np.arange(npoint)%N
        return torch.from_numpy(x).unsqueeze(0)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    if oldidx.shape[1]<=npoint:
        centroids[:,:oldidx.shape[1]]=oldidx[0]
    else:
        centroids[:, :npoint] = oldidx[0,:npoint]
        return centroids
    distance = torch.ones(B, N).to(device) * 1e10
    oldxyz = xyz[:,oldidx[0],:] #[1,M,3]
    distance = xyz[:,None,:,:] - oldxyz[:,:,None,:]
    distance = torch.sum(distance**2,dim=-1)
    distance = torch.min(distance,dim=1)[0]
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # farthest = torch.zeros((B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(oldidx.shape[1],npoint):
        # centroids[:, i] = farthest
        # centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # dist = torch.sum((xyz - centroid) ** 2, -1)
        # mask = dist < distance
        # distance[mask] = dist[mask]
        #
        farthest = torch.max(distance, -1)[1]
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
    return centroids

def PointToLineSegmentDis(P,A,B):
    #点到线段的距离， 点的size为1，线段时多条
    AB = B - A
    AB_norm = AB / (torch.linalg.norm(AB,dim=-1, keepdim=True)+1e-8)

    PA = A - P[None,:]
    BP = P[None,:] - B
    s = torch.multiply(PA,AB_norm).sum(-1)
    t = torch.multiply(BP, AB_norm).sum(-1)
    z = torch.zeros(s.shape[0]).to(P.device).float()

    tmp = torch.vstack([s,t,z])
    h = torch.max(tmp,dim=0)[0]

    AP = P - A
    cross_result = torch.cross(AB_norm,AP,dim=-1)
    c = torch.linalg.norm(cross_result,dim=-1)

    return torch.sqrt(c**2+h**2)


def farthest_point_sampleV3(xyz, npoint):
    """
    基于线段的采样， 用的是线段到线段的距离
    定义为 两个端点到另一条线段的点到线段距离的最大值
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
        oldidx: [1,M]
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    seg = xyz.view(1,-1,2,3)
    device = xyz.device
    B, N, _, C = seg.shape
    if npoint>=N:
        x = np.arange(npoint)%N
        return torch.from_numpy(x).unsqueeze(0)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_line = seg[batch_indices,farthest,:,:]

        # p = np.array([[0, 0, 0]])
        # a = np.array([[1, 1, 0],
        #               [-1, 0, 0],
        #               [-1, -1, 0],
        #               [1, 3, 0]])
        # b = np.array([[2, 2, 0],
        #               [1, 0, 0],
        #               [1, -1, 0],
        #               [1, 2, 0]])
        # ss = PointToLineSegmentDis(torch.from_numpy(p).float()[0], torch.from_numpy(a).float(), torch.from_numpy(b).float())
        # print(ss)
        distA= PointToLineSegmentDis(centroid_line[0,0],seg[0,:,0,:],seg[0,:,1,:])
        distB = PointToLineSegmentDis(centroid_line[0, 1], seg[0, :, 0, :], seg[0, :, 1, :])
        dist = torch.maximum(distA,distB).unsqueeze(0)


        # centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    centroids = torch.cat([centroids*2,centroids*2+1],dim=0)
    centroids = centroids.view(1,-1)
    return centroids




def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    ret_mask = mask[0].sum(-1)/mask.shape[-1]
    ret_mask = ret_mask<0.5
    group_idx[mask] = group_first[mask]
    return group_idx, ret_mask

def query_ball_pointV2(radius, nsample, xyz, new_xyz):
    # 区别在于sort了一下距离
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    sorted_sqrdists, arg_sqrdists = torch.sort(sqrdists, dim=-1)

    arg_sqrdists[sorted_sqrdists > radius ** 2] = N
    arg_sqrdists = arg_sqrdists[:,:,:nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = (arg_sqrdists == N)
    arg_sqrdists[mask] = group_first[mask]
    ret_mask = mask[0].sum(-1)/mask.shape[-1]
    ret_mask = ret_mask<0.5

    return arg_sqrdists, ret_mask

def pointSegDis(xyz, xyzSeg):
    # 点到直线距离
    #xyzSeg : B,N,2,3
    # xyz: B,S,3
    _,S,_=xyz.shape
    _,N,_,_=xyzSeg.shape
    AB = xyzSeg[:,:,1,:] - xyzSeg[:,:,0,:] # B,N,3
    P = xyz.unsqueeze(2).repeat((1,1,N,1)) # B,S,1,3
    A = xyzSeg[:,:,0,:].unsqueeze(1).repeat((1,S,1,1)) # B,1,N,3
    AP = P-A
    AB1 = AB.unsqueeze(1).repeat((1,S,1,1)) # B,S,N,3
    AB_norm = torch.linalg.norm(AB,dim=-1) # B,N
    y = torch.cross(AB1,AP,dim=-1) # B,S,N,3
    y = torch.linalg.norm(y,dim=-1) # B,S,N
    AB_norm = AB_norm.unsqueeze(1).repeat((1,S,1))
    dis = y/(AB_norm+1e-6)

    D = torch.ones_like(dis).float()*100000
    dis = torch.where(AB_norm==0, D, dis)
    return dis

def pointLineSegDis(xyz,xyzSeg):
    # 点到线段的距离
    _,S,_ = xyz.shape # [1,S,3], S=256
    _,N,_,_ = xyzSeg.shape # [1,N,2,3]
    A = xyzSeg[:,:,0,:] # [1,N,3]
    B = xyzSeg[:,:,1,:]
    P = xyz # [1,S,3]
    AB = B - A # [1,N,3]
    AB_norm = AB / (torch.linalg.norm(AB,dim=-1,keepdim=True)+1e-8) #[1,N,3]

    PA = A[:,None,:,:] - P[:,:,None,:]  # [1,S,N,3]
    BP = P[:,:,None,:] - B[:,None,:,:]

    s = torch.multiply(PA,AB_norm.unsqueeze(1)).sum(-1)
    t = torch.multiply(BP, AB_norm.unsqueeze(1)).sum(-1)
    z = torch.zeros_like(s).to(P.device).float()

    tmp = torch.stack([s, t, z], dim=-1)
    h = torch.max(tmp,dim=-1)[0]

    AP = P[:,:,None,:] - A[:,None,:,:]

    AB_norm_repeat = AB_norm.unsqueeze(1).repeat(1,S,1,1)
    cross_result = torch.cross(AB_norm_repeat,AP,dim=-1)
    c = torch.linalg.norm(cross_result,dim=-1)
    return torch.sqrt(c**2+h**2)

def query_ball_pointV3(radius, nsample, xyzSeg, new_xyz, thresh=0.5, sign=None):
    # 区别在于使用点到直线的距离, sort距离 归一化后
    # 的距离在0.2外的都标记为负样本了，然后选前16个， 并且返回前16个中负样本比例超过0.5的mask
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyzSeg: all points, [B, N, 2, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = new_xyz.device
    B,N,_,C = xyzSeg.shape
    B,S,_ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    # dists = pointSegDisV2(new_xyz,xyzSeg) # 点到线段端点距离的最小值
    # dist_pointToLineSegment = pointLineSegDis(new_xyz,xyzSeg)
    # dists = dist_pointToLineSegment
    dists = pointSegDis(new_xyz,xyzSeg)

    sorted_dists, arg_dists = torch.sort(dists, dim=-1)
    arg_dists[sorted_dists > radius] = N #0.2为半径
    arg_dists = arg_dists[:,:,:nsample]
    sorted_dists = sorted_dists[:,:,:nsample] #02月新加，把距离加到xyz坐标后
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = (arg_dists == N)
    arg_dists[mask] = group_first[mask]
    ret_mask = mask[0].sum(-1)/mask.shape[-1]
    ret_mask = ret_mask<thresh
    return arg_dists, ret_mask, sorted_dists, mask


def query_ball_pointV5(radius, nsample, xyzSeg, new_xyz, thresh=0.5):
    # 区别在于使用点到直线的距离, sort距离 归一化后
    # 的距离在0.2外的都标记为负样本了，然后选前16个， 并且返回前16个中负样本比例超过0.5的mask
    #  计算两种距离然后返回其中一种作为特征
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyzSeg: all points, [B, N, 2, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = new_xyz.device
    B,N,_,C = xyzSeg.shape # B,N,2,3
    B,S,_ = new_xyz.shape # B,256,3
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    dist_pointToLineSegment = pointLineSegDis(new_xyz,xyzSeg) #点到线段距离
    dist_pointToLine = pointSegDis(new_xyz,xyzSeg) # 点到线距离

    dists = dist_pointToLine #zheli
    sorted_dists, arg_dists = torch.sort(dists, dim=-1)

    ret_dist = dist_pointToLineSegment #zheli
    tmp = torch.arange(S).unsqueeze(-1).repeat(1,N)
    ret_dist = ret_dist[0,tmp,arg_dists[0]].unsqueeze(0)
    ret_dist = ret_dist[:,:,:nsample]#02月新加，把距离加到xyz坐标后

    arg_dists[sorted_dists > radius] = N #0.2为半径
    arg_dists = arg_dists[:,:,:nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = (arg_dists == N)
    arg_dists[mask] = group_first[mask]
    ret_mask = mask[0].sum(-1)/mask.shape[-1]
    ret_mask = ret_mask<thresh
    return arg_dists, ret_mask, ret_dist, mask



def farthest_point_sampleV4(xyz, npoint):
    """
    这个应该是基于角度进行FPS
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
        oldidx: [1,M]
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    def dist(center,other):
        pass
    seg = xyz.view(1,-1,2,3)
    device = xyz.device
    B, N, _, C = seg.shape
    if npoint>=N:
        x = np.arange(npoint)%N
        return torch.from_numpy(x).unsqueeze(0)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) * 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    seg_dir = seg[:,:,0,:] - seg[:,:,1,:]
    seg_dir_norm = seg_dir / torch.linalg.norm(seg_dir,dim=-1,keepdims=True)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_line = seg[batch_indices,farthest,:,:]

        centroid_line_dir = centroid_line[:,0,:] - centroid_line[:,1,:]
        centroid_line_norm = centroid_line_dir / torch.linalg.norm(centroid_line_dir,dim=-1,keepdims=True)

        dist = (centroid_line_norm[:,None,:] * seg_dir_norm).sum(-1)
        dist = 1 - torch.abs(dist)
        # print(ss)
        # distA= PointToLineSegmentDis(centroid_line[0,0],seg[0,:,0,:],seg[0,:,1,:])
        # distB = PointToLineSegmentDis(centroid_line[0, 1], seg[0, :, 0, :], seg[0, :, 1, :])
        # dist = torch.maximum(distA,distB).unsqueeze(0)


        # centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    # centroids = torch.cat([centroids*2,centroids*2+1],dim=0)
    # centroids = centroids.view(1,-1)
    return centroids



def query_ball_pointV6(radius, nsample, xyzSeg, new_xyz, thresh=0.5):
    # 区别在于使用点到线的距离, 归一化后  的距离在0.2外的都标记为负样本了，然后选前32个， 并且返回前16个中负样本比例超过0.5的
    # 在V3基础上做了一些调整， 计算两种距离然后返回其中一种作为特征
    # 先选靠近的 nsample*2个, 然后按FPS采样32个， FPS基于的是线段之间的角度(余弦距离)
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyzSeg: all points, [B, N, 2, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = new_xyz.device
    B,N,_,C = xyzSeg.shape # B,N,2,3
    B,S,_ = new_xyz.shape # B,256,3
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])


    dist_pointToLine = pointSegDis(new_xyz, xyzSeg)  # 点到线距离
    dist_pointToLineSegment = pointLineSegDis(new_xyz, xyzSeg)  # 点到线段距离

    dists = dist_pointToLine #zheli
    sorted_dists, arg_dists = torch.sort(dists, dim=-1)
    ret_dist = dist_pointToLineSegment #zheli


    # arg_dists[sorted_dists > radius] = N #0.2为半径
    arg_dists = arg_dists[:,:,:2*nsample] # 选出前64个

    zzz = xyzSeg[0,arg_dists[0],:,:]
    out = []
    for kkk in range(zzz.shape[0]):
        result = farthest_point_sampleV4(zzz[kkk],nsample)
        out.append(result)
    out = torch.vstack(out)
    Ind1 = torch.arange(out.shape[0]).view(out.shape[0], 1).repeat((1, out.shape[1]))
    arg_dists_out2 = arg_dists[:,Ind1,out]   #这里包括了一个在64维中的index 到原始线云数量index的转换


    #这个时重新采样出来的 然后需要根据距离mask一下
    sorted_dists_out2 = dists[0,Ind1,arg_dists_out2[0]].unsqueeze(0)
    ret_sorted_dist_out2 = ret_dist[0,Ind1,arg_dists_out2[0]].unsqueeze(0)
    arg_dists_out2[sorted_dists_out2 > radius] = N

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = (arg_dists_out2 == N)
    arg_dists_out2[mask] = group_first[mask]
    ret_mask = mask[0].sum(-1)/mask.shape[-1]
    ret_mask = ret_mask<thresh
    return arg_dists_out2, ret_mask,ret_sorted_dist_out2 ,mask


def query_ball_pointV7(radius, nsample, xyzSeg, new_xyz, thresh=0.5):
    # 区别在于使用点到线的距离, 归一化后  的距离在0.2外的都标记为负样本了，然后选前32个， 并且返回前16个中负样本比例超过0.5的
    # 在V3基础上做了一些调整， 计算两种距离然后返回其中一种作为特征
    # 先选靠近的 nsample*2个, 然后按FPS采样32个， FPS基于的是线段之间的角度(余弦距离)， （已测试效果不佳)
    # 按角度采样8个 按原来的点到直线距离采样24个 (可能重复）
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyzSeg: all points, [B, N, 2, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = new_xyz.device
    B,N,_,C = xyzSeg.shape # B,N,2,3
    B,S,_ = new_xyz.shape # B,256,3
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])


    dist_pointToLine = pointSegDis(new_xyz, xyzSeg)  # 点到线距离
    dist_pointToLineSegment = pointLineSegDis(new_xyz, xyzSeg)  # 点到线段距离

    dists = dist_pointToLine #zheli
    sorted_dists, arg_dists = torch.sort(dists, dim=-1)
    ret_dist = dist_pointToLineSegment #zheli


    # arg_dists[sorted_dists > radius] = N #0.2为半径
    arg_dists = arg_dists[:,:,:2*nsample] # 选出前64个

    zzz = xyzSeg[0,arg_dists[0],:,:]
    out = []
    for kkk in range(arg_dists.shape[1]):
        result = farthest_point_sampleV4(zzz[kkk],8)
        out.append(result)
    out = torch.vstack(out)
    Ind1 = torch.arange(out.shape[0]).view(out.shape[0], 1).repeat((1, out.shape[1]))
    arg_dists_out2 = arg_dists[:,Ind1,out]   #这里包括了一个在64维中的index 到原始线云数量index的转换

    arg_dists_out_temp = arg_dists[:,:,:24]

    arg_dists_out2 = torch.cat([arg_dists_out2,arg_dists_out_temp],dim=-1)


    #这个时重新采样出来的 然后需要根据距离mask一下
    Ind1 = torch.arange(arg_dists_out2[0].shape[0]).view(arg_dists_out2[0].shape[0], 1).repeat((1, arg_dists_out2[0].shape[1]))
    sorted_dists_out2 = dists[0,Ind1,arg_dists_out2[0]].unsqueeze(0)
    ret_sorted_dist_out2 = ret_dist[0,Ind1,arg_dists_out2[0]].unsqueeze(0)
    arg_dists_out2[sorted_dists_out2 > radius] = N

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = (arg_dists_out2 == N)
    arg_dists_out2[mask] = group_first[mask]
    ret_mask = mask[0].sum(-1)/mask.shape[-1]
    ret_mask = ret_mask<thresh
    return arg_dists_out2, ret_mask,ret_sorted_dist_out2 ,mask


def min_angle_to_main_direction(main_dir,seg,nsample=32):
    # main_dir: 256,3,3, seg:256,64,2,3
    seg = seg[:,:,0,:] - seg[:,:,1,:] # 256,64,3
    seg_norm = seg / torch.linalg.norm(seg,dim=-1,keepdim=True) #256,64,3
    # seg_norm = seg_norm.unsqueeze(2) # 256,64,1,3

    dot = main_dir[:,None,:,:] * seg_norm[:,:,None,:] #256,64,3,3
    dot = torch.sum(dot,dim=-1) # 256,64,3
    dot = torch.abs(dot) # no direction angle
    e = 1e-8
    dot = torch.clip(dot,min=e,max=1-e)
    dot = torch.arccos(dot) # 256,64,3
    dot = torch.min(dot,dim=-1)[0] # min of three main direction 256,64
    return dot


def query_ball_pointV8(radius, nsample, xyzSeg, new_xyz, thresh=0.5):
    # 区别在于使用点到线的距离, 归一化后  的距离在0.2外的都标记为负样本了，然后选前32个， 并且返回前16个中负样本比例超过0.5的
    # 在V3基础上做了一些调整， 计算两种距离然后返回其中一种作为特征
    # 按点到线距离采样2*nsample个， 然后基于主方向 采样
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyzSeg: all points, [B, N, 2, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = new_xyz.device
    B,N,_,C = xyzSeg.shape # B,N,2,3
    B,S,_ = new_xyz.shape # B,256,3
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    # xxx = selectedSeg.view(-1,selectedSeg.shape[2]*2,3)
    # xxx = xxx - fpsPoint[0].unsqueeze(1)
    # n=selectedSeg.shape[2]*2
    # C = torch.eye(n) - 1/n * torch.ones((n,n))
    # C = C.unsqueeze(0).repeat((xxx.shape[0],1,1))
    # Scat = torch.bmm(xxx.permute(0,2,1),C)
    # Scat = torch.bmm(Scat,xxx)
    # U,sig,_ = torch.linalg.svd(Scat)
    # ccc = sig[:,1]/sig[:,0]
    # classifyLabel = torch.where(classifyLabel==1,ccc,classifyLabel.float())
    # classifyLabel = classifyLabel/torch.max(classifyLabel)

    dist_pointToLine = pointSegDis(new_xyz, xyzSeg)  # 点到线距离
    dist_pointToLineSegment = pointLineSegDis(new_xyz, xyzSeg)  # 点到线段距离

    dists = dist_pointToLine #zheli
    sorted_dists, arg_dists = torch.sort(dists, dim=-1)
    ret_dist = dist_pointToLineSegment #zheli


    # arg_dists[sorted_dists > radius] = N #0.2为半径
    arg_dists = arg_dists[:,:,:2*nsample] # 选出前64个 (1,256,64)
    arg_lines = xyzSeg[:,arg_dists[0],:,:] # (1, group_numbers, 2 * nsample, 2, 3)
    select_top_lines = arg_lines[0] # (group_numbers, 2*nsample,2,3)
    arg_lines = arg_lines - new_xyz.unsqueeze(2).unsqueeze(2) #(1, group_numbers, 2* nsample, 2,3)
    arg_lines = arg_lines[0] # (group_numbers,2*nsample,2,3)
    arg_lines = arg_lines.view(arg_lines.shape[0],-1,3) # (grouP_numbers, 2*nsample*2, 3)
    n = arg_lines.shape[1]
    C = torch.eye(n) - 1/n * torch.ones((n,n))
    C = C.unsqueeze(0).repeat((arg_lines.shape[0],1,1))
    Scat = torch.bmm(arg_lines.permute(0,2,1),C)
    Scat = torch.bmm(Scat,arg_lines)
    U,sig,_ = torch.linalg.svd(Scat)
    # U: (256,3,3)
    # input:U , select_top_lines, output 32lines
    # 距离是 到主方向三条线的夹角的最小值

    degree_dists = min_angle_to_main_direction(U,select_top_lines) # (1,256,64)
    degree_dists = degree_dists.unsqueeze(0)
    sorted_degree_dists,arg_degree_dists = torch.sort(degree_dists,dim=-1)
    arg_degree_dists = arg_degree_dists[:,:,:nsample] # (1,256,32)
    #要把64内的index 转化为原始的index
    Ind1 = torch.arange(arg_dists.shape[1]).view(arg_dists.shape[1],1).repeat((1,arg_degree_dists.shape[2])) #(256,32)
    global_arg_degree_dists = arg_dists[:,Ind1,arg_degree_dists[0]]

    global_sorted_dists = dists[:,Ind1,global_arg_degree_dists[0]]
    ret_sorted_dists = ret_dist[:,Ind1,global_arg_degree_dists[0]]
    global_arg_degree_dists[global_sorted_dists>radius] = N

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = (global_arg_degree_dists == N)
    global_arg_degree_dists[mask] = group_first[mask]
    ret_mask = mask[0].sum(-1)/mask.shape[-1]
    ret_mask = ret_mask<thresh
    return global_arg_degree_dists,ret_mask,ret_sorted_dists,mask

    zzz = xyzSeg[0,arg_dists[0],:,:]
    out = []
    for kkk in range(arg_dists.shape[1]):
        result = farthest_point_sampleV4(zzz[kkk],8)
        out.append(result)
    out = torch.vstack(out)
    Ind1 = torch.arange(out.shape[0]).view(out.shape[0], 1).repeat((1, out.shape[1]))
    arg_dists_out2 = arg_dists[:,Ind1,out]   #这里包括了一个在64维中的index 到原始线云数量index的转换

    arg_dists_out_temp = arg_dists[:,:,:24]

    arg_dists_out2 = torch.cat([arg_dists_out2,arg_dists_out_temp],dim=-1)


    #这个时重新采样出来的 然后需要根据距离mask一下
    Ind1 = torch.arange(arg_dists_out2[0].shape[0]).view(arg_dists_out2[0].shape[0], 1).repeat((1, arg_dists_out2[0].shape[1]))
    sorted_dists_out2 = dists[0,Ind1,arg_dists_out2[0]].unsqueeze(0)
    ret_sorted_dist_out2 = ret_dist[0,Ind1,arg_dists_out2[0]].unsqueeze(0)
    arg_dists_out2[sorted_dists_out2 > radius] = N

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = (arg_dists_out2 == N)
    arg_dists_out2[mask] = group_first[mask]
    ret_mask = mask[0].sum(-1)/mask.shape[-1]
    ret_mask = ret_mask<thresh
    return arg_dists_out2, ret_mask,ret_sorted_dist_out2 ,mask



def pointSegDisV2(xyz, xyzSeg):
    # 返回点到线段两个端点的最小值
    #xyzSeg : B,N,2,3
    # xyz: B,S,3
    _,S,_=xyz.shape
    _,N,_,_=xyzSeg.shape
    dist = ((xyz.unsqueeze(2).unsqueeze(2) - xyzSeg.unsqueeze(1))**2).sum(-1)
    dist = torch.sqrt(dist)
    dist = torch.min(dist,dim=-1)[0]
    return dist



def query_ball_pointV4(radius, nsample, xyzSeg, new_xyz):
    # 使用点到线段的端点距离的最小值
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyzSeg: all points, [B, N, 2, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = new_xyz.device
    B,N,_,C = xyzSeg.shape
    B,S,_ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    dists = pointSegDisV2(new_xyz,xyzSeg)
    sorted_dists, arg_dists = torch.sort(dists, dim=-1)

    arg_dists[sorted_dists > radius] = N
    arg_dists = arg_dists[:,:,:nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = (arg_dists == N)
    arg_dists[mask] = group_first[mask]
    ret_mask = mask[0].sum(-1)/mask.shape[-1]
    ret_mask = ret_mask<0.5
    return arg_dists, ret_mask, sorted_dists, mask


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def preprocess(junc3D_array):
    R = 6
    dist = np.linalg.norm(junc3D_array[:, None, :] - junc3D_array[None, :, :], axis=-1)
    AA = (dist <= R).sum(-1)
    selectIdx = (AA > 10).nonzero()[0]

    confi = AA[selectIdx].tolist()
    xyz = junc3D_array[selectIdx].tolist()
    all = list(zip(xyz,confi,selectIdx.tolist()))
    all = sorted(all,key=lambda x:x[1],reverse=True)
    xyz,confi,selectIdx = zip(*all)
    xyz = np.array(xyz,dtype=np.float64)

    dropped_junc_index = []
    nms_threshhold = R/3
    for j in range(len(all)):
        if j in dropped_junc_index:
            continue
        dist_all = np.linalg.norm(xyz - xyz[j], axis=1)
        same_region_indexes = (dist_all < nms_threshhold).nonzero()
        for same_region_index_i in same_region_indexes[0]:
            if same_region_index_i == j:
                continue
            if confi[same_region_index_i] <= confi[j]:
                dropped_junc_index.append(same_region_index_i)
            else:
                dropped_junc_index.append(j)
    all = list(zip(xyz.tolist(),confi,selectIdx))
    all = [all[k] for k in range(len(all)) if k not in dropped_junc_index]
    xyz,confi,selectIdx = zip(*all)
    return selectIdx

def preprocessV2(junc3D_array,R=6):
    seg3D_array = junc3D_array.reshape(-1,2,3)
    # R = 6
    junc3D_array = junc3D_array.astype(np.float32)
    dist = np.linalg.norm(junc3D_array[:, None, :] - junc3D_array[None, :, :], axis=-1)
    AA = (dist <= R).sum(-1)
    BB = (dist <= R/2).sum(-1)
    CC = BB/(AA+1e-3)
    # selectIdx = (AA > 12).nonzero()[0]
    selectIdx = ((AA>12) & (CC>0.7)).nonzero()[0]
    if selectIdx.shape[0]==0:
        return np.array([])
    confi = CC[selectIdx].tolist()
    xyz = junc3D_array[selectIdx].tolist()
    all = list(zip(xyz,confi,selectIdx.tolist()))
    all = sorted(all,key=lambda x:x[1],reverse=True)
    xyz,confi,selectIdx = zip(*all)
    xyz = np.array(xyz,dtype=np.float64)

    dropped_junc_index = []
    nms_threshhold = R/4
    for j in range(len(all)):
        if j in dropped_junc_index:
            continue
        dist_all = np.linalg.norm(xyz - xyz[j], axis=1)
        same_region_indexes = (dist_all < nms_threshhold).nonzero()
        for same_region_index_i in same_region_indexes[0]:
            if same_region_index_i == j:
                continue
            if confi[same_region_index_i] <= confi[j]:
                dropped_junc_index.append(same_region_index_i)
            else:
                dropped_junc_index.append(j)
    all = list(zip(xyz.tolist(),confi,selectIdx))
    all = [all[k] for k in range(len(all)) if k not in dropped_junc_index]
    xyz,confi,selectIdx = zip(*all)
    return selectIdx

def preprocessV4(junc3D_array,R=6):
    #专门测试real data的
    seg3D_array = junc3D_array.reshape(-1,2,3)
    # R = 6
    junc3D_array = junc3D_array.astype(np.float32)
    dist = np.linalg.norm(junc3D_array[:, None, :] - junc3D_array[None, :, :], axis=-1)
    AA = (dist <= R).sum(-1)
    BB = (dist <= R/2).sum(-1)
    CC = BB/(AA+1e-3)
    # selectIdx = (AA > 12).nonzero()[0]
    selectIdx = ((AA>12) & (CC>0.7)).nonzero()[0]
    if selectIdx.shape[0]==0:
        return np.array([])
    confi = CC[selectIdx].tolist()
    xyz = junc3D_array[selectIdx].tolist()
    all = list(zip(xyz,confi,selectIdx.tolist()))
    all = sorted(all,key=lambda x:x[1],reverse=True)
    xyz,confi,selectIdx = zip(*all)
    xyz = np.array(xyz,dtype=np.float64)

    dropped_junc_index = []
    nms_threshhold = 4
    for j in range(len(all)):
        if j in dropped_junc_index:
            continue
        dist_all = np.linalg.norm(xyz - xyz[j], axis=1)
        same_region_indexes = (dist_all < nms_threshhold).nonzero()
        for same_region_index_i in same_region_indexes[0]:
            if same_region_index_i == j:
                continue
            if confi[same_region_index_i] <= confi[j]:
                dropped_junc_index.append(same_region_index_i)
            else:
                dropped_junc_index.append(j)
    all = list(zip(xyz.tolist(),confi,selectIdx))
    all = [all[k] for k in range(len(all)) if k not in dropped_junc_index]
    xyz,confi,selectIdx = zip(*all)
    return selectIdx


def preprocessV2_para(junc3D_array,R=6,point_in_circle=12,half_thresh=0.7):
    seg3D_array = junc3D_array.reshape(-1,2,3)
    dist = np.linalg.norm(junc3D_array[:, None, :] - junc3D_array[None, :, :], axis=-1)
    AA = (dist <= R).sum(-1)
    BB = (dist <= R/2).sum(-1)
    CC = BB/(AA+1e-3)
    # selectIdx = (AA > 12).nonzero()[0]
    selectIdx = ((AA>point_in_circle) & (CC>half_thresh)).nonzero()[0]
    if selectIdx.shape[0]==0:
        return np.array([])
    confi = CC[selectIdx].tolist()
    xyz = junc3D_array[selectIdx].tolist()
    all = list(zip(xyz,confi,selectIdx.tolist()))
    all = sorted(all,key=lambda x:x[1],reverse=True)
    xyz,confi,selectIdx = zip(*all)
    xyz = np.array(xyz,dtype=np.float64)

    dropped_junc_index = []
    nms_threshhold = R/4
    for j in range(len(all)):
        if j in dropped_junc_index:
            continue
        dist_all = np.linalg.norm(xyz - xyz[j], axis=1)
        same_region_indexes = (dist_all < nms_threshhold).nonzero()
        for same_region_index_i in same_region_indexes[0]:
            if same_region_index_i == j:
                continue
            if confi[same_region_index_i] <= confi[j]:
                dropped_junc_index.append(same_region_index_i)
            else:
                dropped_junc_index.append(j)
    all = list(zip(xyz.tolist(),confi,selectIdx))
    all = [all[k] for k in range(len(all)) if k not in dropped_junc_index]
    xyz,confi,selectIdx = zip(*all)
    return selectIdx


def preprocessV3(junc3D_array):
    #先采样一些密集地方的点， 然后换算成线， 做线nms
    seg3D_array = junc3D_array.reshape(-1,2,3)
    R = 6
    dist = np.linalg.norm(junc3D_array[:, None, :] - junc3D_array[None, :, :], axis=-1)
    AA = (dist <= R).sum(-1)
    BB = (dist <= R/2).sum(-1)
    CC = BB/(AA+1e-3)
    # selectIdx = (AA > 12).nonzero()[0]
    selectIdx = ((AA>10) & (CC>0.7)).nonzero()[0]
    if selectIdx.shape[0]==0:
        return np.array([])
    confi = CC[selectIdx].tolist()
    xyz = junc3D_array[selectIdx].tolist()
    all = list(zip(xyz,confi,selectIdx.tolist()))
    all = sorted(all,key=lambda x:x[1],reverse=True)
    xyz,confi,selectIdx = zip(*all)
    xyz = np.array(xyz,dtype=np.float64)

    dropped_junc_index = []
    nms_threshhold = R/4
    for j in range(len(all)):
        if j in dropped_junc_index:
            continue
        dist_all = np.linalg.norm(xyz - xyz[j], axis=1)
        same_region_indexes = (dist_all < nms_threshhold).nonzero()
        for same_region_index_i in same_region_indexes[0]:
            if same_region_index_i == j:
                continue
            if confi[same_region_index_i] <= confi[j]:
                dropped_junc_index.append(same_region_index_i)
            else:
                dropped_junc_index.append(j)
    all = list(zip(xyz.tolist(),confi,selectIdx))
    all = [all[k] for k in range(len(all)) if k not in dropped_junc_index]
    xyz,confi,selectIdx = zip(*all)

    selectSegIdx = np.array(selectIdx)//2
    selectSeg = seg3D_array[selectSegIdx,:,:]

    dist = line_to_line_dist(selectSeg,selectSeg)

    dropped_junc_index = []
    nms_threshhold = R
    for j in range(dist.shape[0]):
        if j in dropped_junc_index:
            continue
        dist_all = dist[j]
        same_region_indexes = (dist_all < nms_threshhold).nonzero()
        for same_region_index_i in same_region_indexes[0]:
            if same_region_index_i == j or dist_all[same_region_index_i]<0.1:
                continue
            else:
                dropped_junc_index.append(same_region_index_i)
    selectIdx = [kk for kk in range(len(selectIdx)) if kk not in dropped_junc_index]
    # all = list(zip(xyz.tolist(),confi,selectIdx))
    # all = [all[k] for k in range(len(all)) if k not in dropped_junc_index]

    return selectIdx


def line_to_line_dist(x,y):
    # x,y, (N,2,3) (S,2,3)
    dis = ((x[:, None, :, None] - y[:, None]) ** 2).sum(-1)
    dis = np.sqrt(dis)
    dis = np.minimum(
        dis[:, :, 0, 0] + dis[:, :, 1, 1], dis[:, :, 0, 1] + dis[:, :, 1, 0]
    )
    return dis


def afterPreprocess(fpsPointIdxALL,junc3D, seg3D, std):
    def idxLocalToGlobal(segJuncIdx,selectedSegIdx):
        xxx = segJuncIdx[selectedSegIdx,:]
        return xxx.view(-1)
    juncIdx = torch.from_numpy(np.arange(junc3D.shape[1]))
    segJuncIdx = juncIdx.view(-1,2)
    # dist = torch.sum((junc3D[0][:,None,:] - junc3D[0][None,:,:])**2,dim=-1)

    data = torch.arange(0,junc3D.shape[1]).unsqueeze(0)
    data_sample,_ =query_ball_pointV3(0.2, 16, seg3D, junc3D)

    fpsPointALL = index_points(junc3D, fpsPointIdxALL)
    i=0
    # for i in range(fpsPointALL.shape[1]):
    while fpsPointIdxALL.shape[1]<128:
        fpsPoint = fpsPointALL[:,i:i+1,:]
        selectedSegIdx = data_sample[:,fpsPointIdxALL[0,i],:]
        bs_view = torch.zeros_like(selectedSegIdx)
        selectedSeg = seg3D[bs_view,selectedSegIdx.long(),:] # (1,sample_points_number,16,2,3)

        globalIdx =idxLocalToGlobal(segJuncIdx,selectedSegIdx.reshape(-1))
        old_local_fpsPointIdx = torch.arange(0,fpsPoint.shape[1]*32,32).unsqueeze(0)
        new_fpsPointIdx = farthest_point_sampleV2(selectedSeg.view(1,-1,3), 3, old_local_fpsPointIdx)
        new_fpsPointIdx = globalIdx[new_fpsPointIdx[0]].view(1, -1)
        new_fpsPoint = index_points(junc3D,new_fpsPointIdx)

        dist = torch.sqrt(torch.sum((new_fpsPoint[0][:,None,:] - fpsPointALL[0][None,:,:])**2,dim=-1))*std
        mask = (torch.min(dist,dim=-1)[0]>3)
        new_fpsPointIdx = torch.masked_select(new_fpsPointIdx,mask.unsqueeze(0))

        fpsPointIdxALL = torch.cat([fpsPointIdxALL,new_fpsPointIdx.unsqueeze(0)],dim=1)
        fpsPointALL = index_points(junc3D, fpsPointIdxALL)
        pass


def angle_line_to_line(X,Y):
    # X: 1,N,2,3
    # Y: 1,S,2,3
    # 计算输入的线段pairwise的夹角， 返回余弦距离
    X,Y = X[0],Y[0]
    X_dir = X[:,0] - X[:, 1] # N,3
    Y_dir = Y[:,0] - Y[:, 1] # S,3
    X_dir_norm = X_dir/torch.linalg.norm(X_dir,dim=-1,keepdim=True)
    Y_dir_norm = Y_dir / torch.linalg.norm(Y_dir, dim=-1, keepdim=True)
    dot = X_dir_norm[:,None,:] * Y_dir_norm[None,:,:]
    dot = torch.sum(dot,dim=-1) # N,S
    dot = torch.abs(dot)
    e = 1e-7
    dot = torch.clip(dot,min=e,max=1-e)
    angle = torch.arccos(dot)
    angle = torch.rad2deg(angle)
    return angle.unsqueeze(0)

def query_ball_point_line_based(radius, nsample, xyzSeg, new_xyz, thresh=0.5):
    # 区别在于使用点到直线的距离, sort距离 归一化后
    # 的距离在0.2外的都标记为负样本了，然后选前16个， 并且返回前16个中负样本比例超过0.5的mask
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyzSeg: all points, [B, N, 2, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    col_nsample = 16
    ptl_nsample = 24
    device = new_xyz.device
    B,N,_,C = xyzSeg.shape
    B,S,_,_ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    dists = angle_line_to_line(new_xyz,xyzSeg) #返回线和线无方向夹角距离  (1,S,N)

    # dists = pointSegDisV2(new_xyz,xyzSeg) # 点到线段端点距离的最小值
    # dist_pointToLineSegment = pointLineSegDis(new_xyz,xyzSeg)
    # dists = dist_pointToLineSegment
    # dists = pointSegDis(new_xyz,xyzSeg)

    sorted_dists, arg_dists = torch.sort(dists, dim=-1)
    arg_dists[sorted_dists > 30] = N #大于30du mask
    arg_dists = arg_dists[:,:,:col_nsample]
    sorted_dists = sorted_dists[:,:,:col_nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, col_nsample])
    mask = (arg_dists == N)
    arg_dists[mask] = group_first[mask]
    ret_mask = mask[0].sum(-1)/mask.shape[-1]
    ret_mask = ret_mask<thresh # 1,S,32

    #再基于点到直线距离各采样一些


    ptl_dist1 = pointSegDis(new_xyz[:, :, 0, :], xyzSeg) # 1,S,N
    ptl_dist2 = pointSegDis(new_xyz[:, :, 1, :], xyzSeg)
    Ind1 = torch.arange(S).view(S,1).repeat(1,arg_dists.shape[2])
    ptl_dist1[:, Ind1, arg_dists[0]] = 10000
    ptl_dist2[:, Ind1, arg_dists[0]] = 10000
    sorted_dists1, arg_dists1 = torch.sort(ptl_dist1, dim=-1)
    sorted_dists2, arg_dists2 = torch.sort(ptl_dist2, dim=-1)

    arg_dists1[sorted_dists1>radius] = N
    arg_dists1 = arg_dists1[:,:,:ptl_nsample]
    sorted_dists1 = sorted_dists1[:,:,:ptl_nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, ptl_nsample])
    mask1 = (arg_dists1==N)
    arg_dists1[mask1] = group_first[mask1]

    arg_dists2[sorted_dists2>radius] = N
    arg_dists2 = arg_dists2[:,:,:ptl_nsample]
    sorted_dists2 = sorted_dists2[:,:,:ptl_nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, ptl_nsample])
    mask2 = (arg_dists2==N)
    arg_dists2[mask2] = group_first[mask2]

    ALL_arg_dists = torch.cat([arg_dists,arg_dists1,arg_dists2],dim=2)
    ALL_sorted_dists = torch.cat([sorted_dists,sorted_dists1,sorted_dists2],dim=2)
    ALL_mask = torch.cat([mask,mask1,mask2],dim=2)

    return ALL_arg_dists,ret_mask,ALL_sorted_dists, ALL_mask
    return arg_dists, ret_mask, sorted_dists, mask
