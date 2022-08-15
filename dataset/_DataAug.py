import numpy as np
import random



def rotate_point_cloud_z(*args):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    # for k in range(batch_data.shape[0]):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])

    return [kk@rotation_matrix.T for kk in args]

def shift(*args, scale_low=0.8, scale_high=1.25, shift_range=0.1):
    '''
    :param args: N*C
    :return:
    '''
    # scale = np.random.uniform(scale_low, scale_high)
    shift = np.random.uniform(-shift_range, shift_range, (3,))
    ret = [a+shift for a in args]
    return ret

def scale(*args, scale_low=0.8, scale_high=1.25, shift_range=0.1):
    '''
    统一scale
    :param args: N*C
    :return:
    '''
    scale = np.random.uniform(scale_low, scale_high)
    ret = [a*scale for a in args]
    return ret

def scale_by_axis(*args, scale_low=0.8, scale_high=1.25):
    '''
    统一scale
    :param args: N*C
    :return:
    '''
    scale = [np.random.uniform(scale_low, scale_high) for kk in range(3) ]
    scale = np.array(scale).reshape(1,3)
    ret = [a*scale for a in args]
    return ret


def AugV2(*args,cfg=None):
    if random.randint(0,1)==1:
        args = scale(*args)
    else:
        args = scale_by_axis(*args)
    args = rotate_point_cloud_z(*args)
    args = shift(*args,shift_range=0.2)
    return args