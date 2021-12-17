import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from my_dataset import LoadSignalDataset
from torch import optim


class SE(nn.Module):
    def __init__(self, chs, r):
        super(SE, self).__init__()
        self.chs = chs
        self.GlobalPooling = nn.AdaptiveAvgPool1d(1)
        self.FC1 = nn.Linear(chs, chs//r)
        self.FC2 = nn.Linear(chs//r, chs)

    def forward(self, x):
        x = self.GlobalPooling(x)
        x = x.view(-1, self.chs)
        x = self.FC1(x)
        x = F.relu(x, inplace=True)
        x = self.FC2(x)
        x = F.relu(x, inplace=True)
        x = torch.sigmoid(x)
        x = torch.unsqueeze(x, dim=-1)
        return x

class Conv_1(nn.Module):
    def __init__(self, chs_in, chs_out, need_SE=False):
        super(Conv_1, self).__init__()
        self.conv = nn.Conv1d(chs_in, chs_out, kernel_size=1, padding=0, bias=True)
        self.BN = nn.BatchNorm1d(chs_out)
        self.Relu = nn.ReLU(inplace=True)
        self.need_SE = need_SE
        if self.need_SE:
            if chs_out >= 64:
                r = 16
            else:
                r = 4
            self.SE = SE(chs=chs_out, r=r)

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.Relu(x)
        if self.need_SE:
            s = self.SE(x)
            x = x * s
        return x


class Conv_3(nn.Module):
    def __init__(self, dilation, padding, chs_in, chs_out, need_SE=False):
        super(Conv_3, self).__init__()
        self.conv = nn.Conv1d(chs_in, chs_out, kernel_size=3, dilation=dilation, padding=padding, bias=True)
        self.BN = nn.BatchNorm1d(chs_out)
        self.Relu = nn.ReLU(inplace=True)
        self.need_SE = need_SE
        if self.need_SE:
            if chs_out >= 64:
                r = 16
            else:
                r = 4
            self.SE = SE(chs=chs_out, r=r)

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.Relu(x)
        if self.need_SE:
            s = self.SE(x)
            x = x * s
        return x


class SE_Res_CNN(nn.Module):
    def __init__(self):
        super(SE_Res_CNN, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1_1 = Conv_3(dilation=1, padding=1, chs_in=1, chs_out=16, need_SE=True)
        self.Conv1_2 = Conv_3(dilation=2, padding=2, chs_in=1, chs_out=16, need_SE=True)
        self.Conv2_0 = Conv_1(chs_in=1, chs_out=32)
        self.Conv2 = Conv_1(chs_in=32, chs_out=16, need_SE=True)

        self.Conv3_1 = Conv_3(dilation=1, padding=1, chs_in=16, chs_out=32, need_SE=True)
        self.Conv3_2 = Conv_3(dilation=2, padding=2, chs_in=16, chs_out=32, need_SE=True)
        self.Conv4_0 = Conv_1(chs_in=16, chs_out=64)
        self.Conv4 = Conv_1(chs_in=64, chs_out=32, need_SE=True)

        self.Conv5_1 = Conv_3(dilation=1, padding=1, chs_in=32, chs_out=64, need_SE=True)
        self.Conv5_2 = Conv_3(dilation=2, padding=2, chs_in=32, chs_out=64, need_SE=True)
        self.Conv6_0 = Conv_1(chs_in=32, chs_out=128)
        self.Conv6 = Conv_1(chs_in=128, chs_out=64, need_SE=True)

        self.Conv7_1 = Conv_3(dilation=1, padding=1, chs_in=64, chs_out=128, need_SE=True)
        self.Conv7_2 = Conv_3(dilation=2, padding=2, chs_in=64, chs_out=128, need_SE=True)
        self.Conv8_0 = Conv_1(chs_in=64, chs_out=256)
        self.Conv8 = Conv_1(chs_in=256, chs_out=128, need_SE=True)

        self.Conv9_1 = Conv_3(dilation=1, padding=1, chs_in=128, chs_out=256, need_SE=True)
        self.Conv9_2 = Conv_3(dilation=2, padding=2, chs_in=128, chs_out=256, need_SE=True)
        self.Conv10_0 = Conv_1(chs_in=128, chs_out=512)
        self.Conv10 = Conv_1(chs_in=512, chs_out=256, need_SE=True)

        self.Conv11 = Conv_3(dilation=1, padding=1, chs_in=256, chs_out=256, need_SE=True)
        self.Conv12 = Conv_3(dilation=1, padding=1, chs_in=256, chs_out=256, need_SE=True)

        self.FC1 = nn.Linear(256 * 3, 256)
        self.FC2 = nn.Linear(256, 64)
        self.FC3 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, signal):
        shortcut = self.Conv2_0(signal)
        x1 = self.Conv1_1(signal)
        x2 = self.Conv1_2(signal)
        x = torch.cat((x1, x2), dim=1)
        x += shortcut
        x = self.Conv2(x)
        x = self.Maxpool(x)

        shortcut = self.Conv4_0(x)
        x1 = self.Conv3_1(x)
        x2 = self.Conv3_2(x)
        x = torch.cat((x1, x2), dim=1)
        x += shortcut
        x = self.Conv4(x)
        x = self.Maxpool(x)

        shortcut = self.Conv6_0(x)
        x1 = self.Conv5_1(x)
        x2 = self.Conv5_2(x)
        x = torch.cat((x1, x2), dim=1)
        x += shortcut
        x = self.Conv6(x)
        x = self.Maxpool(x)

        shortcut = self.Conv8_0(x)
        x1 = self.Conv7_1(x)
        x2 = self.Conv7_2(x)
        x = torch.cat((x1, x2), dim=1)
        x += shortcut
        x = self.Conv8(x)
        x = self.Maxpool(x)

        shortcut = self.Conv10_0(x)
        x1 = self.Conv9_1(x)
        x2 = self.Conv9_2(x)
        x = torch.cat((x1, x2), dim=1)
        x += shortcut
        x = self.Conv10(x)
        x = self.Maxpool(x)

        shortcut = x
        x = self.Conv11(x)
        x = self.Conv12(x)
        x += shortcut
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


if __name__ == "__main__":
    epoch = 500
    batch_size = 128
    lr = 0.0005
    lr_unchanged = True
    acc_time = 0
    loss_sum = 0
    valid_score = 0
    min_score = 9999
    shuffle_dataset = True
    train_split = 0.8

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader = LoadSignalDataset("../train_part.csv", need_over_sample=True)
    valid_loader = LoadSignalDataset("../valid.csv", need_over_sample=False)
    train_loader = DataLoader(train_loader, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_loader, batch_size=batch_size, drop_last=True, shuffle=True)

    model = SE_Res_CNN().to(device)
    opt = optim.SGD(model.parameters(), lr=lr)
    Loss = nn.L1Loss(reduction='sum')

    for i in range(0, epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.train(mode=True)
            output = model(data.to(torch.float32))

            opt.zero_grad()
            loss = Loss(output, target)

            loss.backward()
            opt.step()

            acc_time += 1
            loss_sum += loss

            with torch.no_grad():
                if batch_idx % 500 == 0:
                    weight = model.Conv1_1.conv.weight
                    weight_abs = weight.abs()
                    grad_abs = weight.grad.abs()
                    print('\tweight:\t\tMean:\t{}, Max:\t{}, Min:\t{}'.format(weight_abs.mean(),
                                                                                    weight_abs.max(),
                                                                                    weight_abs.min()))
                    print('\tgrad:\t\tMean:\t{}, Max:\t{}, Min:\t{}'.format(grad_abs.mean(),
                                                                                  grad_abs.max(),
                                                                                  grad_abs.min()))

                    for (valid_data, labels) in valid_loader:
                        valid_data, labels = valid_data.to(device), labels.to(device)
                        model.eval()
                        valid_output = model(valid_data.to(torch.float32))
                        valid_score += Loss(valid_output, labels)

                    print('epoch:{}, batch:{}, loss:{}, valid_loss:{}'.format(i + 1, batch_idx, loss_sum / acc_time, valid_score))

                    if lr_unchanged and valid_score < 1000:
                        opt = optim.SGD(model.parameters(), lr=lr/5)
                        lr_unchanged = False

                    if valid_score < 1000 and valid_score < min_score:
                        min_score = valid_score
                        print('Saving SE_Res_CNN.pth')
                        torch.save(model, 'SE_Res_CNN.pth')

                    acc_time = 0
                    loss_sum = 0
                    valid_score = 0

        if i % 100 == 49:
            print('Saving SE_Res_CNN-{}.pth'.format(i+1))
            torch.save(model, 'SE_Res_CNN-{}.pth'.format(i+1))