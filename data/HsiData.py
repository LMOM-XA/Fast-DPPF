import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
from scipy.io import loadmat
from config import DefaultConfig
from collections import deque

import utile.generalUtile

opt = DefaultConfig


#  正则化还没写
class HsiData:
    def __init__(self, data_name, model, train_set_proportion, disjointed, number_threshold):
        '''

        :param data_name:数据集名称
        :param model:ODPPF或PPF
        :param train_set_proportion:训练集占比
        :param disjointed:'random'表示随机划分, 'block'表示按区块disjointed划分，'topBottom'表示从上至下每个类别选取去一半作为训练集。
        :param number_threshold:是
        '''

        data_root = opt.origin_data_root[data_name]
        label_root = opt.origin_label_data_root[data_name]
        hsi_data = loadmat(data_root[0])[data_root[1]]
        hsi_label = loadmat(label_root[0])[label_root[1]]
        shape = hsi_data.shape

        self.origin_data = hsi_data.astype('float32')
        self.origin_label = hsi_label.astype('int64')
        self.hsi_shape = shape

        # 这里的xxx_set 里面的元素代表训练集测试集与验证集.(此处存放的是在origin_data中的下标[x, y])
        self.train_set = []
        self.test_set = []
        self.valid_set = []

        # 配对后的数据（下标）
        self.train_set_paired = []
        self.train_label_paired = []  # 与上一行对应的label

        # 背景点的类别
        self.background_label = 0

        # 正则化
        self.regularization()
        # 得到划分好的训练集与测试集
        self.get_divided_data(train_set_proportion, disjointed, model, number_threshold)

    def regularization(self):
        # 正则化
        shape = self.origin_data.shape
        data_temp = self.origin_data.reshape(-1, shape[-1])
        data_temp = StandardScaler().fit_transform(data_temp)
        data_temp = data_temp.reshape(shape)
        self.origin_data = data_temp

    # disjointed_tr_te()的子函数
    def init_sign(self):
        """
        对processed_sign进行初始化
        :return:
        """

        # 是否已经算进广度算法里的标号，背景类别标号直接初始化为0，其余1。[145,145]
        processed_sign = np.zeros([self.hsi_shape[0], self.hsi_shape[1]], dtype='int32')

        hsi_position = np.array(np.where(self.origin_label != self.background_label),
                                dtype='int32')  # 记录那些非0 样本的位置（写在这里确实有点突兀，但是写在下面不太方便）
        hsi_position = hsi_position.T  # 对它进行转置，这样hsi_position[0]就可以代表第0个点的位置（x,y）。

        for xy in hsi_position:
            x, y = xy
            processed_sign[x][y] = 1

        return processed_sign

    # disjointed_tr_te()的子函数
    def divide_cluster_label(self, processed_sign):
        """
        根据不同的连通集划分出聚类号 cluster_label
        """

        # cluster_label[0]表示类别，cluster_label[1]表示聚类号。[2,145,145]
        cluster_label = np.zeros([2, self.hsi_shape[0], self.hsi_shape[1]], dtype='int32')
        cluster_label[0] = self.origin_label

        # self.background_label在openmax为True时是0
        # self.background_label在openmax为False时是-1
        bc = self.background_label
        label_num = np.max(self.origin_label)  # 类标的最大值
        # label_num = max(self.hsi_label)
        label_list = np.arange(bc + 1, label_num + 1)  # HSI中类别，除了背景点
        cluster_num = 0  # 聚标识类号

        for row in range(self.hsi_shape[0]):
            for column in range(self.hsi_shape[1]):

                label = self.origin_label[row][column]
                sign = processed_sign[row][column]

                # 选定需要聚类的点，若不在聚类范围之内，则跳过
                if (label in label_list) and (sign == 1):
                    cluster_num += 1  # 聚类号要改变

                    # one_label_cluster = np.array([row, column])  # 存放一块相同类的列表,先把此点加进去之后找出的点都拼接于此
                    breadth_first_que = deque()  # 广度优先队列
                    point = np.array([row, column])  # 要进行搜索的点[x,y]

                    breadth_first_que.append(point)  # 初始化广度优先队列
                    processed_sign[row][column] = 0  # 标号的改变

                    while len(breadth_first_que) != 0:
                        # 弹出队首，并记录队首的cluster
                        point = breadth_first_que.popleft()
                        x, y = point
                        cluster_label[1][x][y] = cluster_num

                        # 用广度优先算法把周围点计入队列
                        self.breadth_first(x, y, breadth_first_que, label, processed_sign)
        return cluster_label

    # disjointed_tr_te()的子函数
    def breadth_first(self, row, column, breadth_first_que, label, processed_sign):
        """
        广度搜索，把row, column周围的点进行广度搜索，搜索过的点在self.processed_sign[x][y]进行置零操作
        :param processed_sign:
        :param row:
        :param column:
        :param breadth_first_que:
        :param label:
        :return:
        """
        # 邻域为3x3
        x_neighbor = [-1, 0, 1]
        y_neighbor = [-1, 0, 1]

        # 使用此点开始广度优先遍历
        # 此点把周围满足的点都加进去

        for x_bia in x_neighbor:
            for y_bia in y_neighbor:
                x = row + x_bia
                y = column + y_bia

                if x >= self.hsi_shape[0] or y >= self.hsi_shape[1] or x < 0 or y < 0:
                    continue

                if (processed_sign[x][y] == 1) and (self.origin_label[x][y] == label):
                    # 放入广度队列 好进行之后遍历
                    point = np.array([x, y])
                    breadth_first_que.append(point)
                    processed_sign[x][y] = 0

    # 把HSI按区块(block)划分为两不相交的部分train和test。
    def disjointed_tr_te_block(self):
        """
        返回不相交的训练集与测试集的下标 train_position, test_position
        :return:
        """

        processed_sign = self.init_sign()
        cluster_label = self.divide_cluster_label(processed_sign)

        shaper = self.hsi_shape

        # 先申请足够的内存。
        train_position = np.zeros([shaper[0] * shaper[1], 2])
        train_num = 0  # 记数

        test_position = np.zeros([shaper[0] * shaper[1], 2])
        test_num = 0  # 记数

        for block_num in np.unique(cluster_label[1]):
            # 0是背景点
            if block_num == 0:
                continue

            temp = (cluster_label[1] == block_num)  # numpy特性
            hsi_label_one_block = self.origin_label[temp]

            position_one_block = np.where(cluster_label[1] == block_num)
            position_one_block = np.array(position_one_block).T  # 把position转换为 (x,y=position[i])

            num_one_block = len(hsi_label_one_block)
            train_num_one_block = int(num_one_block / 2)
            test_num_one_block = num_one_block - train_num_one_block

            next_train_num = train_num + train_num_one_block
            next_test_num = test_num + test_num_one_block

            train_position[train_num:next_train_num] = position_one_block[0:train_num_one_block]
            test_position[test_num:next_test_num] = position_one_block[train_num_one_block:]

            train_num = next_train_num
            test_num = next_test_num

        train_position = train_position[0:train_num]
        test_position = test_position[0:test_num]

        train_set = train_position.astype('int32')
        test_set = test_position.astype('int32')

        # np.save('train_position.npy', train_position)
        # np.save('test_position.npy', test_position)
        return train_set, test_set

    # 把HSI按从上之下的顺序(top to bottom)划分为两不相交的部分train和test。
    def disjointed_tr_te_topBottom(self):

        # 先找出每个类别有多少个数据
        hsi_label = self.origin_label.reshape(-1)
        hsi_label_unBc = hsi_label[hsi_label != self.background_label]  # 排除掉背景点
        label_list = np.unique(hsi_label_unBc)  # 每个类别的class
        labels_num = np.zeros(len(label_list))  # 每个类别的数量
        for i, label in enumerate(label_list):
            label_num = np.sum(hsi_label_unBc == label)
            labels_num[i] = label_num

        # 根据每个类别的数量，从上至下，划分出一半作为训练集，另一半作为测试集
        train_set = []
        test_set = []

        for i, label in enumerate(label_list):
            train_count = int(labels_num[i] / 2)  # 每个类别训练集的数量
            num = 0  # 记录已经加入训练集的数量
            for x in range(self.hsi_shape[0]):
                for y in range(self.hsi_shape[1]):

                    # 没超过训练集的数据就归为训练集
                    if label == self.origin_label[x][y] and num < train_count:
                        train_set.append(np.array([x, y]))
                        num += 1

                    # 超过就归为测试集
                    if label == self.origin_label[x][y] and num >= train_count:
                        test_set.append(np.array([x, y]))
                        num += 1

        train_set = np.array(train_set)
        test_set = np.array(test_set)

        np.save('train_set.npy', train_set)
        np.save('test_set.npy', test_set)

        return train_set, test_set

    # 有关ODPPF
    def group_class(self, train_set):
        """
        把train_set中同类别放在一个group中
        :param train_set:
        :return:
        """
        train_label = np.zeros(len(train_set), dtype='int64')
        for i, (x, y) in enumerate(train_set):
            train_label[i] = self.origin_label[x][y]
        label_list = np.unique(train_label)

        # 把相同的类别放在一个组里，（里面放的是下标）
        same_label_group = []
        for label in label_list:
            temp_position = []  # 单独一个类别的
            for (x, y) in train_set:
                if self.origin_label[x][y] == label:
                    temp_position.append((x, y))

            same_label_group.append(temp_position)

        return same_label_group

    # 有关ODPPF
    def get_each_pixel_distance(self, position_group):
        """
        计算position里每个点之间的距离。
        思路是先计算第一个点与其他所有点的距离，存入list。同理计算其他所有点的距离，每个点存入一个list。最后输出一个包含所有list的大list
        :param position_group: 传入的hsi像素点的位置信息
        :return:一个包含position中所有点之间的距离。类型为list。
        """

        distance_list = []  # 记录的是所有点的距离信息

        data_length = len(position_group)  # 这里的position_group 是一个Numpy类型，但是len函数也可以正确计算其长度
        for i in range(data_length):
            position_A = position_group[i]  # 所要计算的像素A的位置
            distance_of_A = []  # 存储A像素对于其余任意像素的距离，之后append进all_distance
            for j in range(data_length):
                position_B = position_group[j]
                distance = utile.generalUtile.get_distance_of_two_pixel(position_A, position_B)
                distance_of_A.append(distance)

            distance_list.append(distance_of_A)

        return distance_list

    # 有关ODPPF (make_pair的子函数)
    def group_class_distance(self, same_label_group):
        """
        计算每个类别中，每个点到其他点之间的距离
        :param same_label_group:
        :return:
        """

        group_distance = []  # 计算完后是一个包含C个组的大list，每个list里面包含一个组之间所有list之间的距离的小list。小list的每一行代表，其中一个点到其余所有点之间的距离。
        for i in range(len(same_label_group)):
            # 计算每一个group（也即同类别class)中，每一个pixel到其他pixel的距离
            one_group_distance = self.get_each_pixel_distance(same_label_group[i])
            group_distance.append(one_group_distance)

        return group_distance

    # ODPPF中使用距离限制来配对
    def make_pixel_pair_ODPPF(self, one_label_group, one_group_distance, number_threshold):
        """
        :param one_label_group: 其中一个类别的下标位置
        :param one_group_distance: 其中一个类别中，每个点到其他点的距离 ne_group_distance[i]对应的就是 one_label_group[i]这个点到其他所有点的距离
        :param number_threshold: ODPPF中选取的配对数量
        :return:
        """

        data_paired = []  # 记录配对了的数据（目前放的是下标）
        label_paired = []

        data_length = len(one_label_group)

        # 距离算法中使用 固定配对数量作为阈值
        for i in range(data_length):  # 此处的i代表是第i个pixel
            # 下面几行是用于使用距离时数据时
            pixel_i_distance = one_group_distance[i]  # 在所有点的距离之中，取出其中一个点对其他所有点的距离
            # np.argsort(x) 输出是x值中 从小到大值的索引。如输出为[5,1,2]，代表x最小值x[5],第二小值为x[1].......
            distance_sorted_index = np.argsort(pixel_i_distance)
            # 取出其中前opt.number_threshold小的索引
            # (例index = distance_sorted_index[2], pixel_i_distance[index] 就是pixel_i_distance中第三小的距离)
            threshold_index = distance_sorted_index[0:number_threshold]

            for j in threshold_index:
                positionA = one_label_group[i]  #
                positionB = one_label_group[j]
                pixel_pair_temp = np.array([positionA, positionB])  # 放的是配对像素的位置
                xA, yA = positionA
                label_temp = self.origin_label[xA][yA]  # 因为是同一类别，只用随便放一个就行

                data_paired.append(pixel_pair_temp)
                label_paired.append(label_temp)

        # 返回配对好的数据（下标）至self.train_set_paired。
        train_set_paired = np.array(data_paired, dtype="int32")
        train_label_paired = np.array(label_paired, dtype='int64')  # 在反向传播中，label必须为long类型也即int64
        return train_set_paired, train_label_paired

    # PPF中全局配对
    def make_pixel_pair_PPF(self, same_label_group):
        """
        same_label_group中的数据的每一个类别进行一一配对
        :param same_label_group:
        :return:
        """

        data_paired = []
        label_paired = []
        # label_class是同一类别的集合,(目前内涵的是下标)
        for same_label_one_class in same_label_group:
            for positionA in same_label_one_class:
                for positionB in same_label_one_class:
                    xA, yA = positionA
                    # xB, yB = positionB
                    pixel_pair_temp = np.array([positionA, positionB])  # 放的是配对像素的位置
                    label_temp = self.origin_label[xA][yA]  # 因为是同一类别，只用随便放一个就行

                    data_paired.append(pixel_pair_temp)
                    label_paired.append(label_temp)
        return data_paired, label_paired

    def make_pair(self, model, train_set, dif_class_num, number_threshold=0):
        """
        对训练集进行一一配对
        :param model: ODPPF或DPPF，他们的配对方式不一样
        :param train_set: 需要配对的训练集（此处存储的是位置[x,y]）
        :param dif_class_num: 影响随机选取0类样本点数量的参数，0类样本点要和其他样本点数量平衡
        :param number_threshold: ODPPF中选取的配对数量
        :return:
        """

        #### 先分组#####
        # 把相同的类别放在一个组里
        # same_label_group[i]存放的是类别为i的所有下标
        same_label_group = self.group_class(train_set)

        #### 配对###
        if model == 'ODPPF':
            if number_threshold == 0:
                print("没有给ODPPF限制配对数量HsiData.py中343行")

            label_0_num = dif_class_num * number_threshold  # 关系到随机选取数量的多少，尽量和每一类平衡

            # 如果是按距离配对（ODPPF）则还得计算距离
            # group_distance[i]存放的是，类别为i的每一个点，到其他类别为i点的距离。
            # group_distance[i][i] 存放的是，类别为i的第一个点，到类别为i的所有点的距离
            group_distance = self.group_class_distance(same_label_group)

            # 使用距离为阈值，执行配对算法。
            # 进行每类 类内配对。
            data_paired = []  # 存放所有配对后的data
            label_paired = []
            for i in range(len(same_label_group)):
                # 进行配对操作
                one_group_paired_data, one_group_paired_label = self.make_pixel_pair_ODPPF(same_label_group[i],
                                                                                           group_distance[i],
                                                                                           number_threshold)

                data_paired.extend(one_group_paired_data)
                label_paired.extend(one_group_paired_label)

        else:  # elif model == 'PPF':
            label_0_num = dif_class_num ** 2  # 关系到随机选取数量的多少，尽量和每一类平衡
            # 同一类别下的像素相互配对
            data_paired, label_paired = self.make_pixel_pair_PPF(same_label_group)

        ###### 随机选取一定数量不同类别的配对 ######
        # 先打乱原始训练集，再进行随机配对，需要保证随机选取出的点不是同类就行
        train_random = utile.generalUtile.random_list(self.train_set, opt.rand_state)
        train_random = np.array(train_random, dtype="int32")
        num_temp = 0
        while num_temp < label_0_num:
            temp_num1 = random.randint(0, len(train_random) - 1)
            temp_num2 = random.randint(0, len(train_random) - 1)

            positionA = train_random[temp_num1]
            positionB = train_random[temp_num2]

            xA, yA = positionA
            xB, yB = positionB

            if self.origin_label[xA][yA] != self.origin_label[xB][yB]:
                pixel_pair_temp = np.array([positionA, positionB])  # 放的是配对像素的位置
                data_paired.append(pixel_pair_temp)
                label_paired.append(0)
                num_temp += 1

        # 返回配对好的数据（下标）至self.train_set_paired。
        train_set_paired = np.array(data_paired, dtype="int32")
        train_label_paired = np.array(label_paired, dtype='int64')  # 在反向传播中，label必须为long类型也即int64
        return train_set_paired, train_label_paired

    def divide_label_unlabeled(self):
        # 这里还需要删除IP的无用点
        # 划分有标记与无标记

        hsi_position_labeled = np.array(np.where(self.origin_label != self.background_label),
                                        dtype='int32')  # 记录那些非0 样本的位置
        train_set = hsi_position_labeled.T  # 对它进行转置，这样hsi_position[0]就可以代表第0个点的位置（x,y）。

        hsi_position_unlabeled = np.array(np.where(self.origin_label == self.background_label),
                                          dtype='int32')  # 记录那些0 样本的位置
        unlabel_set = hsi_position_unlabeled.T  # 对它进行转置，这样hsi_position[0]就可以代表第0个点的位置（x,y）。

        return train_set

    def divide_set_percent(self, train_set, percent):
        '''
        集合划分, 用位置代替实际的集合，节省内存使用
        位置是在Patch数组中的下标
        :return:
        '''

        # 划分测试集，训练集，验证集

        train_set, test_set = train_test_split(train_set, test_size=(1 - percent),
                                               random_state=opt.rand_state, shuffle=True)

        # self.train_set = np.array(self.train_set)
        # self.test_set = np.array(test_set)

        return train_set, test_set

    def divide_set_num(self, train_set, num):
        """
        :param train_set: 需要划分的训练集（以位置的形式给出[x.y]）
        :param num:
        :return:
        """

        train_set_return = []
        test_set_return = []

        # 划分训练集与测试集
        train_label = np.zeros(len(train_set), dtype='int64')
        for i, (x, y) in enumerate(train_set):
            train_label[i] = self.origin_label[x][y]

        for label_name in np.unique(train_label):  # label_name 一个类别的label

            # 这里取出此label的所有序列
            temp_label_list = []
            for (x, y) in train_set:
                label = self.origin_label[x][y]
                if label == label_name:
                    temp_label_list.append((x, y))

            # 这里是打乱顺序
            temp_label_list = utile.generalUtile.random_list(temp_label_list, rstate=opt.rand_state)
            # 下面是打乱顺序后，按照打乱的顺序组装。
            for cont, l in enumerate(temp_label_list):  # l: label
                if cont < num:
                    train_set_return.append(l)
                else:
                    test_set_return.append(l)

        # self.train_set = np.array(train_set_return)
        # self.test_set = np.array(test_set_return)

        train_set = np.array(train_set_return)
        test_set = np.array(test_set_return)

        return train_set, test_set

    def divide_validation(self, test_set, val_set_proportion):
        valid_set, test_set = train_test_split(test_set, test_size=(1 - val_set_proportion),
                                               random_state=opt.rand_state, shuffle=True)
        # self.valid_set = np.array(self.valid_set)

        return valid_set

    def get_divided_data(self, train_set_proportion, disjointed, model, number_threshold):

        """
        对HSI数据进行划分，选取出配对后的训练集，以及测试集与验证集
        :param train_set_proportion: 训练集占比，大于1时表示取固定个数，为0--1时表示取百分比训练集
        :param disjointed: 'random'表示随机划分, 'block'表示按区块disjointed划分，'topBottom'表示从上至下每个类别选取去一半作为训练集。
        :param model:PPF或ODPPF
        :param number_threshold: ODPPF中的参数，关系ODPPF配对数量
        :return:
        """

        if disjointed == 'block':  # 采用disjointed_block划分时
            train_set, test_set = self.disjointed_tr_te_block()
            # 测试集是来自另一个disjointed block，而不是从训练集中划分的
            if train_set_proportion > 1:
                train_set, _ = self.divide_set_num(train_set, train_set_proportion)
            else:
                train_set, _ = self.divide_set_percent(train_set, train_set_proportion)

        elif disjointed == 'topBottom':  # 采用disjointed top to bottom划分时
            train_set, test_set = self.disjointed_tr_te_topBottom()
            if train_set_proportion > 1:
                train_set, _ = self.divide_set_num(train_set, train_set_proportion)
            else:
                train_set, _ = self.divide_set_percent(train_set, train_set_proportion)

        else:  # 采用随机划分时
            train_set = self.divide_label_unlabeled()
            if train_set_proportion > 1:
                train_set, test_set = self.divide_set_num(train_set, train_set_proportion)
            else:
                train_set, test_set = self.divide_set_percent(train_set, train_set_proportion)

        # 是否启用验证集
        if opt.validation is True:
            val_set = self.divide_validation(test_set, opt.val_set_proportion)
            self.valid_set = val_set

        self.train_set = train_set
        # self.train_set = test_set
        self.test_set = test_set

        # 对训练集进行配对
        self.train_set_paired, self.train_label_paired = self.make_pair(model, train_set, train_set_proportion,
                                                                        number_threshold)
        self.display()

    def display(self):

        labels = self.origin_label.reshape(-1)
        train_labels = np.zeros(len(self.train_set), dtype='int32')
        for i, (x, y) in enumerate(self.train_set):
            train_labels[i] = self.origin_label[x][y]
        test_labels = np.zeros(len(self.test_set), dtype='int32')
        for i, (x, y) in enumerate(self.test_set):
            test_labels[i] = self.origin_label[x][y]

        counter_total = Counter(labels)
        counter_total = sorted(counter_total.items(), key=lambda x: x[0], reverse=False)
        counter_train = Counter(train_labels)
        counter_train = sorted(counter_train.items(), key=lambda x: x[0], reverse=False)
        counter_test = Counter(test_labels)
        counter_test = sorted(counter_test.items(), key=lambda x: x[0], reverse=False)
        counter_paired = Counter(self.train_label_paired)
        counter_paired = sorted(counter_paired.items(), key=lambda x: x[0], reverse=False)

        print('dataset:{}'.format(opt.data_name))
        print('data length:{}'.format(self.hsi_shape[0] * self.hsi_shape[1]))
        print('具体：{}'.format(counter_total))
        print('hsi shape:{}'.format(self.hsi_shape))
        print('测试集大小:{}'.format(len(test_labels)))
        print('测试集为:{}'.format(counter_test))
        print('训练集中:{}'.format(counter_train))
        print('配对后训练集大小为{}'.format(len(self.train_label_paired)))
        print('配对后训练集为{}'.format(counter_paired))
