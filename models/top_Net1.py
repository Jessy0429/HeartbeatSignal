import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, kernel, dilation, chs_in, chs_out):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(chs_in, chs_out, kernel_size=kernel, dilation=dilation, padding=0, bias=True)
        self.BN = nn.BatchNorm1d(chs_out)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.Relu(x)
        return x

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.Conv1 = Conv(kernel=3, dilation=1, chs_in=1, chs_out=16)
        self.Conv2 = Conv(kernel=3, dilation=2, chs_in=16, chs_out=32)
        self.Conv3 = Conv(kernel=3, dilation=2, chs_in=32, chs_out=64)
        self.Conv4 = Conv(kernel=5, dilation=2, chs_in=64, chs_out=64)

        self.Conv5 = Conv(kernel=5, dilation=2, chs_in=64, chs_out=128)
        self.Conv6 = Conv(kernel=5, dilation=2, chs_in=128, chs_out=128)

        self.FC1 = nn.Linear(39 * 128, 256)
        self.FC21 = nn.Linear(39 * 128, 16)
        self.FC22 = nn.Linear(16, 256)
        self.FC3 = nn.Linear(256, 4)

        self.Maxpool0 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.spatial_dropout = nn.Dropout2d(p=0.2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Maxpool1(x)

        x = self.Conv5(x)
        x = self.Conv6(x)
        x = self.Maxpool0(x)

        x = self.spatial_dropout(x)

        x = x.view(-1, 39*128)
        x1 = self.FC1(x)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        x2 = self.FC21(x)
        x2 = F.relu(x2)
        x2 = self.FC22(x2)
        x2 = torch.sigmoid(x2)
        x2 = self.dropout(x2)
        x = self.FC3(x1 + x2)
        x = F.softmax(x, dim=-1)

        return x