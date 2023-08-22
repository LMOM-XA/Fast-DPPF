import numpy as np
from collections import Counter
from data.HsiData import HsiData


# HsiData类的文字可视化
def show_HsiData_by_text(hsi_data):
    # hsi_data = np.array(hsi_data.hsi_data)  # 具体data就不用了
    hsi_label = hsi_data.hsi_label
    hsi_data.hsi_shape
    # hsi_data.hsi_position  # 直接算，应该也不用

    labeled = np.array(hsi_data.labeled)
    hsi_data.unlabeled

    # 这里的xxx_set 里面的元素代表，在hsi_data与hsi_label 中的位置
    train_set = hsi_data.train_set  # 有标记集中分为训练集测试集与验证集.
    test_set = hsi_data.test_set
    hsi_data.valid_set

    pseudo_label = hsi_data.pseudo_label  # 这里是由推理产生的伪标签，从unlabel中出现
    hsi_data.pseudo_label_confidence  # 以及伪标签的置信度
    trusted_set = hsi_data.trusted_set  # 置信达到的伪标签集。

    pseudo_train_set = hsi_data.pseudo_train_set  # 达到置信中的伪标签用来训练部分,是trusted_set的一小部分


    # 显示train和test
    train_label = hsi_label[train_set]
    counter_train = Counter(train_label)
    counter_train = sorted(counter_train.items(), key=lambda x: x[0], reverse=False)
    print('train total:{}'.format(len(train_label)))
    print('train具体：{}'.format(counter_train))

    test_label = hsi_label[test_set]
    counter_test = Counter(test_label)
    counter_test = sorted(counter_test.items(), key=lambda x: x[0], reverse=False)
    print('test total:{}'.format(len(test_label)))
    print('test具体：{}'.format(counter_test))

    # 显示伪标记
    counter_pseudo = Counter(pseudo_label)
    counter_pseudo = sorted(counter_pseudo.items(), key=lambda x: x[0], reverse=False)
    print('pseudo total:{}'.format(len(pseudo_label)))
    print('pseudo具体：{}'.format(counter_pseudo))

    # 显示达到置信的伪标记
    trusted_label = pseudo_label[trusted_set]
    counter_trusted = Counter(trusted_label)
    counter_trusted = sorted(counter_trusted.items(), key=lambda x: x[0], reverse=False)
    print('trusted total:{}'.format(len(trusted_label)))
    print('trusted具体：{}'.format(counter_trusted))

    # 显示达到trust_train_set,也即trusted中用来训练部分
    pseudo_train_label = pseudo_label[pseudo_train_set]
    counter_pseudo_train = Counter(pseudo_train_label)
    counter_pseudo_train = sorted(counter_pseudo_train.items(), key=lambda x: x[0], reverse=False)
    print('pseudo_train total:{}'.format(len(pseudo_train_label)))
    print('pseudo_train具体：{}'.format(counter_pseudo_train))


# HsiData类的图像可视化
def show_HsiData_by_pic(hsi_data):
    pass


# 伪标记数据与原始数据差异的可视化图像
