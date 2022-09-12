
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class classify(nn.Module):
    def __init__(self, mlp):
        super(classify, self).__init__()
        last_channel = mlp[0]
        self.mlp = []
        for out_channel in mlp[1:]:
            self.mlp.append(nn.Conv1d(last_channel,out_channel, 1, 1))  # use conv1D
            if out_channel == mlp[-1]:
                break
            self.mlp.append(nn.BatchNorm1d(out_channel))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(0.1))
            last_channel = out_channel
        self.mlp = nn.Sequential(*self.mlp)
    def forward(self, X):
        X = self.mlp(X)
        return X

