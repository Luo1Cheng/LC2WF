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
import torch.nn as nn
from utils.dup_stdout_manager import DupStdoutFileManager
from dataset import classifyDataset
from modeling import build_model
from utils.utils import optimizer_define, load_model, save_model, AverageMeter
import argparse
import torch.nn.functional as F
from itertools import permutations
import json
def TrainInit(yamlName):
    if yamlName=="":
        pyFileName = os.path.splitext(os.path.basename(__file__))[0]
    else:
        pyFileName = yamlName
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
        optimizer.zero_grad()
        bs = batch['input'].size(0)
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
                # print(batch[k].shape)
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
        print(Bar.suffix)
    ret = {k:v.avg for k,v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret



from valid_code import validCls
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process some string")
    parser.add_argument("--yamlName", type=str, default="", help="input the yaml file name", required=False)
    args = parser.parse_args()
    cfg = TrainInit(args.yamlName)
    log_path = cfg['log_path']
    with DupStdoutFileManager(os.path.join(log_path, 'logfile.txt')) as _:
        print(json.dumps(cfg,indent=4,ensure_ascii=False))
        # --------------------------CPU GPU-----------------------------
        train_dataset = classifyDataset(cfg['dataset'])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['train']['batch_size'],
                                                   shuffle=True, num_workers=cfg['train']['num_workers'],
                                                   pin_memory=True, drop_last=False, collate_fn=train_dataset.collate_fn)

        test_dataset = classifyDataset(cfg['dataset'], split='test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                   shuffle=False, num_workers=cfg['train']['num_workers'],
                                                   pin_memory=True, drop_last=False, collate_fn=train_dataset.collate_fn)
        # --------------------------Model Optimizer Scheduler-----------------------------
        model = build_model(cfg,model="classify")
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
        start_epoch = 0
        end_epoch = cfg['train']['optim_step'][-1]
        best_AP = 0
        if cfg['mode']=="train":
            print("\033[31m#####Training connectivity model begins#####\033[0m")
            for epoch in range(start_epoch + 1, end_epoch + 1):
                print('epoch',epoch)
                t1 = time.time()
                log_dict_train = train(epoch, model_parallel, optimizer, train_loader,end_epoch)
                t2 = time.time()

                scheduler.step(epoch=epoch)
                loss = log_dict_train['loss']
                if (epoch % 2 == 0) or (epoch >= 15) or (epoch == 1):
                    t1 = time.time()
                    this_ap = validCls(model_parallel, test_loader, log_path, cfg)
                    t2 = time.time()
                    print("valid {} sec".format((t2 - t1) / 60))
                    save_model(os.path.join(log_path, 'saved_models', 'model_epoch{}.pth'.format(epoch)), epoch, loss,
                               model)
                    if this_ap > best_AP:
                        best_AP = this_ap
                        save_model(os.path.join(log_path, 'saved_models', "edge_best"), epoch,
                                   loss, model)
        elif cfg['mode']=="eval":
            print("\033[31m#####Predicting connectivity begins#####\033[0m")
            for epoch in range(0, 1):
                validCls(model_parallel, test_loader, log_path, cfg)
        else:
            print("\033[31m#####set the right mode in yaml file#####\033[0m")
