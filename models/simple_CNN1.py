import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, chs_in, chs_out):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(chs_in, chs_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.BN = nn.BatchNorm1d(chs_out)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.Relu(x)
        return x


class Simple_CNN1(nn.Module):
    def __init__(self):
        super(Simple_CNN1, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1 = Conv(chs_in=1, chs_out=16)
        self.Conv2 = Conv(chs_in=16, chs_out=32)
        self.Conv3 = Conv(chs_in=32, chs_out=64)
        self.FC1 = nn.Linear(64*25, 64)
        self.FC2 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, signal):
        x = self.Conv1(signal)
        x = self.Maxpool(x)
        x = self.Conv2(x)
        x = self.Maxpool(x)
        x = self.Conv3(x)
        x = self.Maxpool(x)
        x = x.view(-1, 64*25)
        x = self.FC1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.FC2(x)
        x = F.relu(x)
        x = F.softmax(x, dim=-1)
        return x
