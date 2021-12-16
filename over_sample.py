# MAHAKIL 重采样
# phase 1 : 计算马氏距离，并递减排序
# phase 2 : 分区
# phase 3 : 合成新样本
import numpy as np
import pandas as pd
from collections import Counter
from data_processing import DrawSignal
from imblearn.over_sampling import SMOTE


class MyMahakil(object):
    def __init__(self):
        self.T = []  # 需要生成的缺陷样本数
        self.new = []  # 存放新生成的样本
        # self.signal_fig = DrawSignal(6, 3)

    # 核心方法
    # return : data_new, label_new
    def fit_sample(self, data, label):
        # data : 包含度量信息的样本 数组
        # label : 样本的标签 数组
        label_counter = Counter(label)
        label_num = len(label_counter)
        datas_classed = []
        # label_list = []
        for i in range(0, label_num):
            datas_classed.append([])
            # label_list.append([])

        # 按照label划分数据集
        for i in range(0, label.shape[0]):
            datas_classed[int(label[i])].append(data[i])

        major_class_num = max(len(i) for i in datas_classed)
        for data_classed in datas_classed:
            self.T.append(major_class_num - len(data_classed))

        for label_index, T in enumerate(self.T):
            if T != 0:
                data_np = np.array(datas_classed[label_index])
                # 计算得到马氏距离
                # print("Calculating mahalanobis distance for samples of label{}".format(label_index))
                # d = self.mahalanobis_distance(data_np)
                print("Calculating euclidean distance for samples of label{}".format(label_index))
                d = self.euclidean_distance(data_np)
                # 降序排序
                d.sort(key=lambda x: x[1], reverse=True)
                # 将正例集一分为二
                k = len(d)
                d_index = [d[i][0] for i in range(k)]
                data_sorted = [data_np[i] for i in d_index]
                mid = int(k/2)
                bin1 = [data_sorted[i] for i in range(0, mid)]
                bin2 = [data_sorted[i] for i in range(mid, k)]
                print("Generating {} new samples for label{}".format(T, label_index))
                # 循环迭代生成新样本
                l_ = len(bin1)
                mark = [1, 3, 7, 15, 31, 63, 127, 255]
                p = T / l_
                is_full = True
                g = mark.index([m for m in mark if m > p][0]) + 1
                cluster = 2 ** (g - 1)  # 最后一代的子代个数
                if (T - mark[g-2]*l_) < cluster:
                    # 说明多增加一代，还不如保持少几个的状态
                    is_full = False
                    g -= 1
                    k = 0
                else:
                    k = l_ - round((T - mark[g-2]*l_)/cluster)
                self.generate_new_sample(bin1, bin2, g, l_, k, is_full)
                # self.signal_fig.draw()
                print("Jointing {} new samples".format(len(self.new)))
                data = np.append(data, np.array(self.new), axis=0)
                label = np.append(label, np.full((len(self.new)), label_index), axis=0)
                self.new = []
        return data, label

    def euclidean_distance(self, x):
        x_mean = np.mean(x, axis=0)
        d = []
        for i in range(x.shape[0]):
            d_squre = np.sqrt(np.square(x[i]-x_mean).sum())
            d_tuple = (i, d_squre)
            d.append(d_tuple)
        return d

    def mahalanobis_distance(self, x):
        # x : 数组
        mu = np.mean(x, axis=0)  # 均值
        d = []
        s = np.cov(x.T)
        inv_s = np.linalg.inv(s)
        for i in range(x.shape[0]):
            x_mu = np.atleast_2d(x[i] - mu)
            d_squre = np.dot(np.dot(x_mu, inv_s), np.transpose(x_mu))[0][0]
            d_tuple = (i, d_squre)
            d.append(d_tuple)
        return d

    @staticmethod
    def cov(x):
        # x : 数组
        s = np.zeros((x.shape[1], x.shape[1]))
        mu = np.mean(x, axis=0)  # 均值
        for i in range(x.shape[0]):
            x_xbr = np.atleast_2d(x - mu)
            s_i = np.dot(np.transpose(x_xbr), x_xbr)
            s = s + s_i
        return np.divide(s, x.shape[0])

    # 生成新样本
    def generate_new_sample(self, bin1, bin2, g, l, k, is_full):
        # bin1, bin2 是数组
        # g 遗传的剩余代数
        # l bin1的item数目
        # k 最后一代每个节点需要裁剪的个数
        # is_full 是否溢出，也即最后一代算完，是超出了T，还是未满T
        assert len(bin1) <= len(bin2)
        if g >= 2 or (g == 1 and is_full is False):
            lv_0 = []  # 子代
            for i in range(l):
                # 生成子代
                lv_0.append(np.mean(np.append(np.atleast_2d(bin1[i]), np.atleast_2d(bin2[i]), axis=0), axis=0))
            # self.signal_fig.add_signal(bin1[0], "parent")
            # self.signal_fig.add_signal(bin2[0], "parent")
            # self.signal_fig.add_signal(lv_0[0], "child")
            # self.signal_fig.add_signal(bin1[1], "parent")
            # self.signal_fig.add_signal(bin2[1], "parent")
            # self.signal_fig.add_signal(lv_0[1], "child")
            self.new.extend(lv_0)
            self.generate_new_sample(lv_0, bin1, g-1, l, k, is_full)
            self.generate_new_sample(lv_0, bin2, g-1, l, k, is_full)
        if g == 1 and is_full:
            lv_0 = []  # 子代
            for i in range(l):
                # 生成子代
                lv_0.append(np.mean(np.append(np.atleast_2d(bin1[i]), np.atleast_2d(bin2[i]), axis=0), axis=0))
            del lv_0[-1: (-k-1): -1]
            self.new.extend(lv_0)


class MySmote():
    def __init__(self):
        self.new = []

    def fit_sample(self, data, label):
        # data : 包含度量信息的样本 数组
        # label : 样本的标签 数组
        smote = SMOTE()
        signal, label = smote.fit_resample(data, label)
        return signal, label


if __name__ == '__main__':
    data = pd.read_csv("./train.csv").drop(['id'], axis=1)
    signal = []
    label = data.iloc[:, 1]

    for i in range(0, len(data)):
        signal.append([float(i) for i in data.iloc[i, 0].split(',')])

    over_sample = MySmote()
    signal, label = over_sample.fit_sample(np.array(signal), np.array(label, dtype=int))
