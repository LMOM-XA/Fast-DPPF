import numpy as np
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import DefaultConfig
import data.myDataset as myDataset

opt = DefaultConfig


class DataLoader:
    """
    负责装载进dataloader
    """

    @staticmethod
    def getin_dataloader_train(origin_data, train_set_paired, train_label_paired, shuffle, batch_size):
        """
        把数据与对应的label装载进dataloader
        :param origin_data: 成patch或非patch的整个HSI data信息 shape为（-1， band)
        :param train_set: 训练集（里面包含的是在origin_data的下标) [x,y]
        :param train_label: 训练集的标签train_label[i]，对应的就是train_set[i]在origin_data的标签
        :param batch_size:
        :param shuffle:是否打乱顺序，测试集与验证集为FALSE
        :return: 装载号的dataloader
        """

        data_set = myDataset.HsiDatasetTrainPPF(origin_data, train_set_paired, train_label_paired)
        dataloader = torch.utils.data.DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=shuffle,
            # num_workers=numWorkers
        )

        return dataloader

    @staticmethod
    def getin_dataloader_test(origin_data, origin_label, test_set, patch_size, shuffle, batch_size):
        """
        把数据与对应的label装载进dataloader
        :param origin_data:
        :param origin_label:
        :param test_set:里面包含的是在原始数据中的下标
        :param patch_size:
        :param shuffle:是否打乱顺序，测试集与验证集为FALSE
        :param batch_size:
        :return: 装载号的dataloader
        """

        data_set = myDataset.HsiDatasetTestPPF(origin_data, origin_label, test_set, patch_size=patch_size)
        dataloader = torch.utils.data.DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=shuffle,
            # num_workers=numWorkers
        )

        return dataloader

    @staticmethod
    def get_pic_dataloader(origin_data, origin_label, data_shape, patch_size, shuffle, batch_size):
        """
        因为要预测全图，为了适应代码，我们只需要把test_set变化一下就行。使test_set里包含全图就行
        :param origin_data:
        :param origin_label:
        :param data_shape:
        :param patch_size:
        :param shuffle:是否打乱顺序，测试集与验证集为FALSE
        :param batch_size:
        :return: 装载号的dataloader
        """

        # 使test_set中包含全图的下标
        test_set = np.zeros((data_shape[0] * data_shape[1], 2), dtype='int32')
        count = 0
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                test_set[count] = np.array((i, j), dtype='int32')
                count += 1
        data_set = myDataset.HsiDatasetTestPPF(origin_data, origin_label, test_set, patch_size=patch_size)
        dataloader = torch.utils.data.DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=shuffle,
            # num_workers=numWorkers
        )

        return dataloader