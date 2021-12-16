import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class DrawSignal():
    def __init__(self, row, col):
        self.col = col
        self.row = row
        self.index = 0
        self.signals = []
        self.labels = []

    def add_signal(self, signal, label):
        if self.col * self.row == self.index:
            return False
        else:
            x_list = []
            y_list = []

            for x, y in enumerate(signal):
                x_list.append(int(x))
                y_list.append(float(y))

            self.signals.append((self.index+1, x_list, y_list))
            self.labels.append(label)

            self.index += 1
            # ax = plt.subplot(self.row, self.col, self.index)
            # ax.set_title("label: {}".format(label))
            # plt.plot(x_list, y_list)
            return True

    def draw(self):
        fig = plt.figure(figsize=(self.col * 5, self.row * 3))
        for (index, x, y), label in zip(self.signals, self.labels):
            ax = plt.subplot(self.row, self.col, index)
            ax.set_title(label)
            plt.plot(x, y)
        plt.tight_layout(pad=5, h_pad=5, w_pad=5)
        plt.show()
        plt.close(fig)
        self.index = 0
        self.signals = []
        self.labels = []


if __name__ == "__main__":
    train = pd.read_csv("./train.csv")
    signals = train.iloc[:, 1:3].values

    signal_fig = DrawSignal(6, 3)
    for signal in signals:
        flag = signal_fig.add_signal(signal[0].split(','), int(signal[1]))
        if not flag:
            signal_fig.draw()
            break


    # fft = np.fft.fft(np.array(y_list), 205)
    # fftshift = np.fft.fftshift(fft)
    # amp = abs(fftshift) / len(fft)
    # pha = np.angle(fftshift)
    # fre = np.fft.fftshift(np.fft.fftfreq(205, 1))


    label = Counter(train['label'])
    print(label, len(label))






