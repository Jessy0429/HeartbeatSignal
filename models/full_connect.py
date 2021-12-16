import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from my_dataset import LoadSignalDataset
from torch import optim


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.FC1 = nn.Linear(205, 64)
        self.FC2 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, signal):
        x = signal.view(-1, 205)
        x = self.FC1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.FC2(x)
        x = F.relu(x)
        x = F.softmax(x, dim=-1)
        return x


if __name__ == "__main__":
    epoch = 100
    batch_size = 256
    lr = 0.001
    lr_unchanged = True
    acc_time = 0
    loss_sum = 0
    valid_score = 0
    shuffle_dataset = True
    train_split = 0.8

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader = LoadSignalDataset("../train_part.csv", need_over_sample=True)
    valid_loader = LoadSignalDataset("../valid.csv", need_over_sample=False)
    train_loader = DataLoader(train_loader, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_loader, batch_size=batch_size, drop_last=True, shuffle=True)

    # model = torch.load('broad_CNN-500.pth').to(device)
    model = FC().to(device)
    # model = Simple_CNN().to(device)
    # model = Encoder_Decoder(data_size=1, hidden_size=8, num_layer=4, seq_len=205, batch_size=batch_size).to(device)
    opt = optim.SGD(model.parameters(), lr=lr)
    # opt = optim.Adam(model.parameters(), lr=0.001)
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

            if batch_idx % 1000 == 0:
                weight = model.FC1.weight
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

                acc_time = 0
                loss_sum = 0
                valid_score = 0

        if i != 0 and i % 50 == 0:
            torch.save(model, 'FC-{}.pth'.format(i))