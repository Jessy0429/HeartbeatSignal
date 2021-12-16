import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from my_dataset import LoadSignalDataset
from torch import optim

from models.full_connect import FC
from models.top_Net1 import Net1
from models.complex_CNN import Complex_CNN
from models.Res_CNN import Res_CNN


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

    # signal_data = LoadSignalDataset("./train.csv", need_over_sample=True)
    # dataset_size = len(signal_data)
    # indices = list(range(dataset_size))
    # split = int(np.floor(train_split * dataset_size))
    # if shuffle_dataset:
    #     np.random.seed()
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[:split], indices[split:]
    #
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)
    # train_loader = DataLoader(signal_data, batch_size=batch_size, sampler=train_sampler)
    # valid_loader = DataLoader(signal_data, batch_size=batch_size, sampler=valid_sampler)

    train_loader = LoadSignalDataset("./train_part.csv", need_over_sample=True)
    valid_loader = LoadSignalDataset("./valid.csv", need_over_sample=False)
    train_loader = DataLoader(train_loader, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_loader, batch_size=batch_size, drop_last=True, shuffle=True)

    # model = torch.load('Res_CNN.pth').to(device)
    # model = Net1().to(device)
    # model = Simple_CNN().to(device)
    # model = Complex_CNN().to(device)
    model = Res_CNN().to(device)
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
                        # opt = optim.Adam(model.parameters(), lr=0.0001)
                        lr_unchanged = False

                    if valid_score < 1000 and valid_score < min_score:
                        min_score = valid_score
                        print('Saving Res_CNN.pth')
                        torch.save(model, 'Res_CNN.pth')

                    acc_time = 0
                    loss_sum = 0
                    valid_score = 0

        if i % 100 == 49:
            print('Saving Res_CNN-{}.pth'.format(i+1))
            torch.save(model, 'Res_CNN-{}.pth'.format(i+1))