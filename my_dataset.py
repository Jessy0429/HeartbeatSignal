import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from over_sample import MyMahakil, MySmote


class LoadSignalDataset(Dataset):
    def __init__(self, file_path, need_over_sample=False, test=False, transform=None, target_transform=None):
        self.test = test
        data = pd.read_csv(file_path).drop(['id'], axis=1)

        signal = []
        if not self.test:
            label = data.iloc[:, 1]


        for i in range(0, len(data)):
            signal.append([float(i) for i in data.iloc[i, 0].split(',')])
        if need_over_sample and not self.test:
            print("Over sampling ...")
            over_sample = MyMahakil()
            # over_sample = MySmote()
            self.signal, self.label = over_sample.fit_sample(np.array(signal), np.array(label, dtype=int))
        else:
            self.signal = np.array(signal)
            if not self.test:
                self.label = np.array(label, dtype=int)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.signal.shape[0]

    def __getitem__(self, item):
        if self.test:
            signal = np.expand_dims(self.signal[item], axis=0)
            return torch.from_numpy(signal)
        else:
            # data = self.dataset.iloc[item, :]
            # signal = np.array([float(i) for i in data[0].split(',')])
            # labels = np.zeros(4)
            # labels[int(data[1])] = 1
            # return torch.from_numpy(signal), torch.from_numpy(labels)

            labels = np.zeros(4)
            labels[int(self.label[item])] = 1

            # 单纯的signal
            signal = np.expand_dims(self.signal[item], axis=0)
            return torch.from_numpy(signal), torch.from_numpy(labels)

            # 引入DFT
            # fft = np.fft.fft(np.array(self.signal[item]), 205)
            # fftshift = np.fft.fftshift(fft)
            # amp = np.expand_dims(abs(fftshift) / len(fft), axis=0)
            # pha = np.expand_dims(np.angle(fftshift), axis=0)
            # signal = np.expand_dims(self.signal[item], axis=0)

            # return torch.from_numpy(np.concatenate((signal, amp, pha), axis=0)), torch.from_numpy(labels)


if __name__ == '__main__':
    my_dataset = LoadSignalDataset()
    my_dataset.__getitem__(213)
