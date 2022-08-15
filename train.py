# -*- coding: UTF-8 -*-
import sys
import torch.utils.data
import random
import os
import time
import numpy as np
from progress.bar import Bar
import torch
import yaml
import datetime
from utils.dup_stdout_manager import DupStdoutFileManager
import dataset
from modeling import build_model
from utils.utils import optimizer_define, load_model, save_model, AverageMeter

import torch.nn.functional as F
import json
def TrainInit():
    pyFileName = os.path.splitext(os.path.basename(__file__))[0]
    yamlFileName = "config/{}.yaml".format(pyFileName)
    if not os.path.exists(yamlFileName):
        raise ValueError("{} not found".format(yamlFileName))
    with open(yamlFileName, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    seed = cfg['seed']
    # ---------------------Init log------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = True

    log_path = os.path.join(cfg['log_path'], "train_log", datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    cfg['log_path'] = log_path
    os.system('mkdir -p {}'.format(os.path.join(log_path, 'src')))
    os.system('cp *.py {}'.format(os.path.join(log_path, 'src')))
    os.system('cp -r modeling {}'.format(os.path.join(log_path, 'src')))
    os.system('cp -r utils {}'.format(os.path.join(log_path, 'src')))
    os.system('cp -r config {}'.format(os.path.join(log_path, 'src')))
    os.system('cp -r dataset {}'.format(os.path.join(log_path, 'src')))
    os.system('mkdir -p {}'.format(os.path.join(log_path, 'saved_models')))
    os.system('mkdir -p {}'.format(os.path.join(log_path, 'results')))

    # --------------------------CPU GPU-----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        raise ValueError("cpu version for training is not implemented.")
        sys.exit()
    print('\n <--------------Using device: ', device,"----------------->")
    return cfg

def train(epoch, model, optimizer, data_loader, end_epoch):
    model.train()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(epoch), max=end_epoch)
    avg_loss_stats = None
    for iter_id, batch in enumerate(data_loader):
        # if iter_id<730:
        #     continue
        optimizer.zero_grad()
        bs = batch['input'].size(0)
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()

        output= model(batch)
        loss = output['loss'].mean()
        if avg_loss_stats is None:
            avg_loss_stats = {l: AverageMeter() for l in output.keys() if "loss" in l}

        loss.backward()
        optimizer.step()
        Bar.suffix = '[{0}/{1}]|'.format(iter_id, num_iters)

        for l in avg_loss_stats:
            avg_loss_stats[l].update(output[l].mean().item(), bs)
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        if iter_id%10==0: print(Bar.suffix)
    ret = {k:v.avg for k,v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret

def Acc(logits, target, thresh=0.5):
    prob = F.softmax(logits, dim=1)
    confi = prob[:, 1, :]
    pred = torch.where(prob[:,1,:]>thresh, 1, 0)

    # X = (prob == target)
    X = (pred==target)
    tp = X * target

    recall = torch.sum(tp, dim=-1)/torch.sum(target, dim=-1)
    recall = torch.mean(recall)

    precision = torch.sum(tp, dim=-1) / torch.sum(pred, dim=-1)

    acc = torch.sum(X,dim=-1)/X.shape[1]
    acc = torch.mean(acc)
    return acc, recall, precision, tp, confi

def AccForBCE(logits, target, thresh=0.5):
    prob = torch.sigmoid(logits)
    confi = prob[:, 0, :]
    pred = torch.where(prob[:,0,:]>thresh, 1, 0)

    # X = (prob == target)
    X = (pred==target)
    tp = X * target

    recall = torch.sum(tp, dim=-1)/torch.sum(target, dim=-1)
    recall = torch.mean(recall)

    precision = torch.sum(tp, dim=-1) / torch.sum(pred, dim=-1)

    acc = torch.sum(X,dim=-1)/X.shape[1]
    acc = torch.mean(acc)
    return acc, recall, precision, tp, confi

def L2dis(mask, predXYZ, targetXYZ, max):
    # residual = torch.abs(predXYZ-targetXYZ)  # bs,256,3
    # residual = torch.mean(residual * max, dim=-1) # bs 256
    residual = torch.abs(predXYZ-targetXYZ)
    residual = torch.sum( (residual * max)**2 ,dim=-1)
    residual = torch.sqrt(residual)
    mae = torch.sum(residual * mask, dim=-1) / torch.sum(mask, dim=-1)
    mae = torch.mean(mae, dim=0)
    return mae


from valid_code import juncRecallV2,getsap

def valid(model, data_loader, log_path):
    model.eval()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(epoch), max=end_epoch)
    avg_loss_stats = None
    CONFI_THRESH = 0.0
    S53 = getsap(s=5,nms_threshhold=3,confi_thresh=CONFI_THRESH)
    S33 = getsap(s=3,nms_threshhold=3,confi_thresh=CONFI_THRESH)
    S73 = getsap(s=7, nms_threshhold=3,confi_thresh=CONFI_THRESH)

    S50 = getsap(s=5, nms_threshhold=0.01,confi_thresh=CONFI_THRESH)
    S30 = getsap(s=3, nms_threshhold=0.01,confi_thresh=CONFI_THRESH)
    S70 = getsap(s=7, nms_threshhold=0.01,confi_thresh=CONFI_THRESH)
    tAll = 0
    tstart = time.time()
    with torch.no_grad():
        for iter_id, batch in enumerate(data_loader):
            bs = batch['input'].size(0)
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda()
            t1 = time.time()
            output = model(batch)



            tAll = tAll + time.time() - t1
            logits = output['l1_logits']


            acc, recall, precision, tpMask, confi = Acc(logits, batch['classifyLabel'])
            tpMAE = L2dis(tpMask, output['l1_predXYZ'], output['l1_targetXYZ'], batch['std'])
            tpfnMAE = L2dis(batch['classifyLabel'].bool(), output['l1_predXYZ'], output['l1_targetXYZ'], batch['std'])


            label_rec_loss = juncRecallV2(output['l1_predXYZ'],confi,batch['objGTJunc3D'],batch['mean'],batch['std'],batch['fpsPoint'])
            S53(output['l1_predXYZ'], confi, batch['objGTJunc3D'], batch['mean'], batch['std'], batch['fpsPoint'], batch['rec_label'])
            S33(output['l1_predXYZ'], confi, batch['objGTJunc3D'], batch['mean'], batch['std'], batch['fpsPoint'],batch['rec_label'])
            S73(output['l1_predXYZ'], confi, batch['objGTJunc3D'], batch['mean'], batch['std'], batch['fpsPoint'],batch['rec_label'])

            S50(output['l1_predXYZ'], confi, batch['objGTJunc3D'], batch['mean'], batch['std'], batch['fpsPoint'], batch['rec_label'])
            S30(output['l1_predXYZ'], confi, batch['objGTJunc3D'], batch['mean'], batch['std'], batch['fpsPoint'],batch['rec_label'])
            S70(output['l1_predXYZ'], confi, batch['objGTJunc3D'], batch['mean'], batch['std'], batch['fpsPoint'],batch['rec_label'])

            if cfg['write_pred_junc']:
                out={
                    "mean": batch['mean'].tolist(),
                    "std": batch['std'].tolist(),
                    "predXYZ":output['l1_predXYZ'].tolist(),
                    "targetXYZ":output['l1_targetXYZ'].tolist(),
                    "logits": logits.tolist(),
                    "classifyLabel":batch['classifyLabel'].tolist(),
                    "fpsLabel": batch['fpsLabel'].tolist(),
                    "label": batch['label'].tolist(),
                    "fpsPoint":batch['fpsPoint'].tolist(),
                    "attn1":output['attn1'].tolist(),
                    "attn2":output['attn2'].tolist(),
                    "patch":batch['input'].tolist(),
                    "wireframeJunc": batch['objGTJunc3D'][0].tolist(),
                    "wireframeLine": batch['objLineIdx'][0].tolist(),
                }
                b = json.dumps(out)
                predJuncDir = cfg['write_pred_junc_path']
                os.makedirs(predJuncDir,exist_ok=True)
                fw = open(os.path.join(predJuncDir,batch['name'][0]+".json"),'w')
                fw.write(b)
                fw.close()

            metric={
                'cls_acc_loss':acc,
                'cls_rec_loss':recall,
                'cls_pre_loss':precision,
                "tp_loss":tpMAE,
                "tpfn_loss":tpfnMAE,
                "label_rec_loss":label_rec_loss,
            }
            if avg_loss_stats is None:
                avg_loss_stats = {l: AverageMeter() for l in metric.keys() if "loss" in l}
            Bar.suffix = '[{0}/{1}]|'.format(iter_id, num_iters)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(metric[l].mean().item(), bs)
                # Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, metric[l].mean().item())
            if iter_id%50==0: print(Bar.suffix,batch['name'])

        def ff(a):
            for kk in a.keys():
                a[kk] = round(a[kk],4)
            return a

        def ss(a,b,c):
            d = {}
            for kk in a.keys():
                d[kk] = a[kk]+b[kk]+c[kk]
                d[kk] = round(d[kk]/3,4)
            return d


        print("###### nms=3 + truncate ######")
        a1 = ff(S33.get_RPF())
        a2 = ff(S53.get_RPF())
        a3 = ff(S73.get_RPF())
        am = ss(a1,a2,a3)
        print("without nms only truncate")
        b1 = ff(S30.get_RPF())
        b2 = ff(S50.get_RPF())
        b3 = ff(S70.get_RPF())
        bm = ss(a1,a2,a3)
        print("\n")

        print("""
            before nms:
            vtx min/max/ave/std {},{},{},{}
            sap3, recall3, {}, {},
            sap5, recall5, {}, {},
            sap7, recall7, {}, {},
            ave sap, ave recall, {},{}

            
            after nms, nms thresh=3
            vtx min/max/ave/std {},{},{},{}
            sap3, recall3, {}, {},
            sap5, recall5, {}, {},
            sap7, recall7, {}, {},
            ave sap, ave recall, {},{}

            vtx 
            """.format(b1['pred_min'], b1['pred_max'], b1['pred_ave'], b1['pred_std'],
              b1['AP'], b1['recall'], b2['AP'], b2['recall'], b3['AP'], b3['recall'],bm['AP'],bm['recall'],
              a1['pred_nms_min'], a1['pred_nms_max'], a1['pred_nms_ave'], a1['pred_nms_std'],
              a1['AP'], a1['recall'], a2['AP'], a2['recall'], a3['AP'], a3['recall'], am['AP'], am['recall'],
              ))


        print("\n")

        print("network forward total time {}, total sample {}, ave {}".format(tAll,data_loader.__len__(),tAll/data_loader.__len__()))
        FF = time.time() - tstart
        print("network total time {}, total sample {}, ave {}".format(FF, data_loader.__len__(),FF / data_loader.__len__()))
        print("save tp fp result of nms=3")




if __name__=="__main__":
    cfg = TrainInit()
    log_path = cfg['log_path']
    with DupStdoutFileManager(os.path.join(log_path,'logfile.txt')) as _:
        print(json.dumps(cfg,indent=4,ensure_ascii=False))
        # --------------------------CPU GPU-----------------------------
        train_dataset = dataset.LineDataset(cfg['dataset'])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['train']['batch_size'],
                                                   shuffle=True, num_workers=cfg['train']['num_workers'],
                                                   pin_memory=True, drop_last=True, collate_fn=train_dataset.collate_fn)
        train_loader2 = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                   shuffle=False, num_workers=cfg['train']['num_workers'],
                                                   pin_memory=True, drop_last=True, collate_fn=train_dataset.collate_fn)

        test_dataset = dataset.LineDataset(cfg['dataset'], split='test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                   shuffle=False, num_workers=cfg['train']['num_workers'],
                                                   pin_memory=True, drop_last=True, collate_fn=train_dataset.collate_fn)
        # --------------------------Model Optimizer Scheduler-----------------------------
        model = build_model(cfg)
        if cfg['model']['load_model'] != '':
            if cfg['train']['resume'] == True:
                print("Resume from model {}".format(cfg['model']['load_model']))
                model, current_loss, start_epoch = load_model(model, cfg['model']['load_model'],
                                                              resume=cfg['train']['resume'],
                                                              selftrain=cfg['train']['self_train'])
            else:
                print("Load model from {}".format(cfg['model']['load_model']))
                model = load_model(model, cfg['model']['load_model'], resume=cfg['train']['resume'],
                                   selftrain=cfg['train']['self_train'])
        model.cuda()
        model.train()
        model_parallel = torch.nn.DataParallel(model)
        optimizer = optimizer_define(model, None, cfg['train'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg['train']['optim_step'], gamma=0.1)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # --------------------------Start Training-----------------------------
        print('---------------Starting training...------------------')
        start_epoch=0
        end_epoch = cfg['train']['optim_step'][-1]
        TRAIN = False
        if TRAIN:
            for epoch in range(start_epoch + 1, end_epoch + 1):
                log_dict_train = train(epoch, model_parallel, optimizer, train_loader,end_epoch)
                scheduler.step(epoch=epoch)
                loss = log_dict_train['loss']
                if (epoch%5==0) or (epoch>=0):
                    print("epoch",epoch)
                    valid(model_parallel, test_loader, log_path)
                    save_model(os.path.join(log_path, 'saved_models', 'model_epoch{}.pth'.format(epoch)), epoch, loss,
                               model)
        else:
            for epoch in range(0, 1):
                print("get test dataset junction prediction results")
                valid(model_parallel, test_loader, log_path)
                # print("get train dataset junction prediction results")
                # valid(model_parallel, train_loader2, log_path)
