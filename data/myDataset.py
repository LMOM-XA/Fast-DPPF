'''
用于构建torch.utils.data.dataset中的Dataset
'''
import numpy as np
from torch.utils.data import Dataset
import torch


# 基础Dataset
class BasedDateset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        # 调代码 不写要报错(转换训练时可以用的格式)
        the_return_hsi_data = torch.from_numpy(self.data[index])  # 转为Tensor
        label = self.label[index]

        return the_return_hsi_data, label

    def __len__(self):
        return self.label.size


# 使用HSI类做dataset
class HsiDatasetTrainPPF(Dataset):
    def __init__(self, origin_data, train_set_paired, train_label_paired):
        '''

        :param origin_data: 成patch或非patch的整个HSI data信息 shape为（-1， band)
        :param train_set: 训练集（里面包含的是在origin_data的下标) [x,y]
        '''
        self.origin_data = origin_data
        self.train_set = train_set_paired
        self.train_label = train_label_paired

    def __getitem__(self, index):
        positionA, positionB = self.train_set[index]
        xA, yA = positionA
        xB, yB = positionB
        pixel_one = self.origin_data[xA][yA]
        pixel_two = self.origin_data[xB][yB]
        pixel_pair = np.array([pixel_one, pixel_two])

        label = self.train_label[index]

        return pixel_pair, label

    def __len__(self):
        return len(self.train_label)


# 使用HSI类做dataset
class HsiDatasetTestPPF(Dataset):
    def __init__(self, origin_data, origin_label, test_set, patch_size):
        '''

        :param origin_data: 成patch或非patch的整个HSI data信息 shape为（-1， band)
        :param test_set: 训练集的标签train_label[i]，对应的就是train_set[i]在origin_data的标签
        '''
        self.origin_data = origin_data
        self.origin_label = origin_label
        self.test_set = test_set
        self.patch_size = patch_size

    def __getitem__(self, index):

        '''
        self.test_set存放的是测试集，具体内容是在self.origin_data中的下标[x,y]
        首先把N邻域（3,5,7,9,11）中的像素点与测试点配对，生成N*N数量的像素对。
        :param index:
        :return: N*N数量的像素对，以及测试点对应的label
        在train中得到返回N*N数量的像素对之后需要经过处理才能放入网络。
        '''
        xA, yA = self.test_set[index]
        margin = (self.patch_size - 1) / 2
        shape = self.origin_data.shape
        band = shape[2]

        pixel_pairs = np.zeros((self.patch_size ** 2, 2, band), dtype='float32')
        num_temp = 0
        for xB in np.arange(xA - margin, xA + margin + 1, dtype='int32'):
            for yB in np.arange(yA - margin, yA + margin + 1, dtype='int32'):

                pixelA = self.origin_data[xA][yA]
                # 范围外就先用相同的pixel填充（训练时有训练过相同的pixel配对）
                if xB < 0 or xB >= shape[0] or yB < 0 or yB >= shape[1]:
                    pixelB = pixelA
                else:
                    pixelB = self.origin_data[xB][yB]
                pixel_pairs[num_temp] = np.array([pixelA, pixelB])
                num_temp += 1

        label = self.origin_label[xA][yA]

        return pixel_pairs, label

    def __len__(self):
        return len(self.test_set)
