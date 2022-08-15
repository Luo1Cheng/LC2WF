import torch
from radam import RAdam
def optimizer_define(model, optim_weight, param):
    optim_weight = {
        'back': 1,
        'line': 1,
        'junc': 1,

    }
    learning_rate = param['lr']
    all_bai, all_wei = [], []
    # if param['freeze']:
    #     print("-------------freeze parameters not in descriptor branch-----------------")
    for pname, p in model.named_parameters():
        if 'bias' in pname or 'bn' in pname:
            all_bai += [p]
        else:
            all_wei += [p]
        # if param['freeze'] == True:
            # print("freeze {}".format(pname))
            # p.requires_grad=False
    optimizer = torch.optim.AdamW([
    # optimizer = RAdam([
        {'params': all_wei, 'lr': optim_weight['back'] * learning_rate, 'weight_decay': 1e-5},
        {'params': all_bai, 'lr': optim_weight['back'] * learning_rate, 'weight_decay': 0},
    ]
    )
    return optimizer
    backbone_bai, backbone_wei = [], []
    line_bai, line_wei = [], []
    junc_bai,junc_wei=[],[]
    line_dis_bai,line_dis_wei=[],[]
    junc_dis_bai,junc_dis_wei=[],[]
    centerness_bai,centerness_wei=[],[]
    center_bai,center_wei=[],[]
    center_degree_bai,center_degree_wei=[],[]
    cut_bai,cut_wei=[],[]
    center_offset_bai,center_offset_wei=[],[]
    for pname, p in model.named_parameters():
        if any([pname.startswith(k) for k in ['resnet', 'up','score','res','hg','layer','conv1','bn1','fc']]):
            if 'bias' in pname or 'bn' in pname:
                backbone_bai += [p]
            else:
                backbone_wei += [p]
        elif any([pname.startswith(k) for k in ['head_line_dis']]):
            if 'bias' in pname or 'bn' in pname:
                line_dis_bai += [p]
            else:
                line_dis_wei += [p]
        elif any([pname.startswith(k) for k in ['head_junc_dis']]):
            if 'bias' in pname or 'bn' in pname:
                junc_dis_bai += [p]
            else:
                junc_dis_wei += [p]
        elif any([pname.startswith(k) for k in ['head_line']]):
            if 'bias' in pname or 'bn' in pname:
                line_bai += [p]
            else:
                line_wei+= [p]
        elif any([pname.startswith(k) for k in ['head_junc']]):
            if 'bias' in pname or 'bn' in pname:
                junc_bai += [p]
            else:
                junc_wei+= [p]
        elif any([pname.startswith(k) for k in ['head_centerness']]):
            if 'bias' in pname or 'bn' in pname:
                centerness_bai += [p]
            else:
                centerness_wei+= [p]
        elif any([pname.startswith(k) for k in ['head_center_degree']]):
            if 'bias' in pname or 'bn' in pname:
                center_degree_bai += [p]
            else:
                center_degree_wei+= [p]
        elif any([pname.startswith(k) for k in ['head_center','head_offset']]):
            if 'bias' in pname or 'bn' in pname:
                center_bai += [p]
            else:
                center_wei+= [p]
        elif any([pname.startswith(k) for k in ['line_conv','center_conv']]):
            if 'bias' in pname or 'bn' in pname:
                cut_bai += [p]
            else:
                cut_wei+= [p]
        else:
            print(pname)

    optimizer = torch.optim.Adam([
        {'params': backbone_wei, 'lr': optim_weight['back'] * learning_rate, 'weight_decay': 1e-5},
        {'params': backbone_bai, 'lr': optim_weight['back'] * learning_rate, 'weight_decay': 0},
        {'params': line_wei, 'lr': optim_weight['line'] * learning_rate, 'weight_decay': 1e-5},
        {'params': line_bai, 'lr': optim_weight['line'] * learning_rate, 'weight_decay': 0},
        {'params': line_dis_wei, 'lr': optim_weight['line_dis'] * learning_rate, 'weight_decay': 1e-5},
        {'params': line_dis_bai, 'lr': optim_weight['line_dis'] * learning_rate, 'weight_decay': 0},
        {'params': junc_wei, 'lr': optim_weight['junc'] * learning_rate, 'weight_decay': 1e-5},
        {'params': junc_bai, 'lr': optim_weight['junc'] * learning_rate, 'weight_decay': 0},
        {'params': junc_dis_wei, 'lr': optim_weight['junc_dis'] * learning_rate, 'weight_decay': 1e-5},
        {'params': junc_dis_bai, 'lr': optim_weight['junc_dis'] * learning_rate, 'weight_decay': 0},
         {'params': centerness_wei, 'lr': optim_weight['centerness'] * learning_rate, 'weight_decay': 1e-5},
        {'params': centerness_bai, 'lr': optim_weight['centerness'] * learning_rate, 'weight_decay': 0},
         {'params': center_wei, 'lr': optim_weight['center'] * learning_rate, 'weight_decay': 1e-5},
        {'params': center_bai, 'lr': optim_weight['center'] * learning_rate, 'weight_decay': 0},
         {'params': center_degree_wei, 'lr': optim_weight['degree'] * learning_rate, 'weight_decay': 1e-5},
        {'params': center_degree_bai, 'lr': optim_weight['degree'] * learning_rate, 'weight_decay': 0},
         {'params': cut_wei, 'lr': optim_weight['cut'] * learning_rate, 'weight_decay': 1e-5},
        {'params': cut_bai, 'lr': optim_weight['cut'] * learning_rate, 'weight_decay': 0},
    ]
    )
    return optimizer


def load_model(model, model_path, resume=False, selftrain=False):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if not selftrain: # True if model is not saved by self-training
        print('loaded', model_path)
        state_dict_ = checkpoint
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()

        # check loaded parameters and created modeling parameters
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        return model
    else:
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']

        state_dict = {}
        # convert data_parallal to modeling
        state_dict=state_dict_
        # for k in state_dict_:
        #     if k.startswith('model'):
        #         state_dict[k[6:]] = state_dict_[k]
        #     else:
        #         state_dict[k] = state_dict_[k]
            # if k.startswith('module') and not k.startswith('module_list'):
            #     state_dict[k[7:]] = state_dict_[k]
            # else:
            #     state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()

        # check loaded parameters and created modeling parameters
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)

        if resume:
            if 'current_loss' in checkpoint:
                current_loss = checkpoint['current_loss']
                start_epoch = checkpoint['epoch']
                print('current_loss:', current_loss)
                return model, current_loss, start_epoch
            else:
                return model, None, start_epoch
        else:
            return model

def save_model(path, epoch, loss, model):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict, 'current_loss': loss}
    torch.save(data, path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count