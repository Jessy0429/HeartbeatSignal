import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_1(nn.Module):
    def __init__(self, chs_in, chs_out):
        super(Conv_1, self).__init__()
        self.conv = nn.Conv1d(chs_in, chs_out, kernel_size=1, padding=0, bias=True)
        self.BN = nn.BatchNorm1d(chs_out)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.Relu(x)
        return x


class Conv_3(nn.Module):
    def __init__(self, dilation, padding, chs_in, chs_out):
        super(Conv_3, self).__init__()
        self.conv = nn.Conv1d(chs_in, chs_out, kernel_size=3, dilation=dilation, padding=padding, bias=True)
        self.BN = nn.BatchNorm1d(chs_out)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.Relu(x)
        return x


class Complex_CNN(nn.Module):
    def __init__(self):
        super(Complex_CNN, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1_1 = Conv_3(dilation=1, padding=1, chs_in=1, chs_out=16)
        self.Conv1_2 = Conv_3(dilation=2, padding=2, chs_in=1, chs_out=16)
        self.Conv2 = Conv_1(chs_in=32, chs_out=16)

        self.Conv3_1 = Conv_3(dilation=1, padding=1, chs_in=16, chs_out=32)
        self.Conv3_2 = Conv_3(dilation=2, padding=2, chs_in=16, chs_out=32)
        self.Conv4 = Conv_1(chs_in=64, chs_out=32)

        self.Conv5_1 = Conv_3(dilation=1, padding=1, chs_in=32, chs_out=64)
        self.Conv5_2 = Conv_3(dilation=2, padding=2, chs_in=32, chs_out=64)
        self.Conv6 = Conv_1(chs_in=128, chs_out=64)

        self.Conv7_1 = Conv_3(dilation=1, padding=1, chs_in=64, chs_out=128)
        self.Conv7_2 = Conv_3(dilation=2, padding=2, chs_in=64, chs_out=128)
        self.Conv8 = Conv_1(chs_in=256, chs_out=128)

        self.Conv9_1 = Conv_3(dilation=1, padding=1, chs_in=128, chs_out=256)
        self.Conv9_2 = Conv_3(dilation=2, padding=2, chs_in=128, chs_out=256)
        self.Conv10 = Conv_1(chs_in=512, chs_out=512)

        self.Conv11 = Conv_3(dilation=1, padding=1, chs_in=512, chs_out=256)
        self.Conv12 = Conv_3(dilation=1, padding=1, chs_in=256, chs_out=256)

        self.FC1 = nn.Linear(256 * 3, 256)
        self.FC2 = nn.Linear(256, 64)
        self.FC3 = nn.Linear(64, 4)
        self.spatial_dropout = nn.Dropout2d(p=0.2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, signal):
        x1 = self.Conv1_1(signal)
        x2 = self.Conv1_2(signal)
        x = torch.cat((x1, x2), dim=1)
        x = self.spatial_dropout(x)
        x = self.Conv2(x)
        x = self.Maxpool(x)

        x1 = self.Conv3_1(x)
        x2 = self.Conv3_2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.spatial_dropout(x)
        x = self.Conv4(x)
        x = self.Maxpool(x)

        x1 = self.Conv5_1(x)
        x2 = self.Conv5_2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.spatial_dropout(x)
        x = self.Conv6(x)
        x = self.Maxpool(x)

        x1 = self.Conv7_1(x)
        x2 = self.Conv7_2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.spatial_dropout(x)
        x = self.Conv8(x)
        x = self.Maxpool(x)

        x1 = self.Conv9_1(x)
        x2 = self.Conv9_2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.spatial_dropout(x)
        x = self.Conv10(x)
        x = self.Maxpool(x)

        x = self.Conv11(x)
        x = self.Conv12(x)
        x = self.Maxpool(x)

        x = x.view(-1, 256 * 3)
        x = self.FC1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.FC2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.FC3(x)
        x = F.relu(x)
        x = F.softmax(x, dim=-1)
        return x
