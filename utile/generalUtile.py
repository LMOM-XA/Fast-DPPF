from collections import Counter

from config import DefaultConfig
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

opt = DefaultConfig


# 调节lr的，每30个epoch降低一会
def adjust_learning_rate(optimizer, epoch):
    # lr = opt.lr * (0.1 ** (epoch // 30)) * (0.1 ** (epoch // 60))
    lr = opt.lr
    if epoch > 100:
        lr *= 0.1
    opt.lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_acc_loss(loss, outputs, labels):
    batch_size = labels.size(0)
    _, train_predicted = torch.max(outputs.data, 1)
    running_loss = loss.data.item() * batch_size
    train_correct = torch.sum(train_predicted == labels.data)

    # return running_loss, train_correct.cpu().numpy()
    return running_loss, train_correct.cpu().numpy()


# 把多维列表拉直成1维并转成numpy
def stretch_list(l):
    temp_list = []

    for element in l:
        temp_list += element.tolist()

    result = np.array(temp_list)
    return result


def get_distance_of_two_pixel(position_pixel_A, position_pixel_B):
    '''
    使用相应的距离算法，计算两个点之间的距离
    :param position_pixel_A:所要计算的像素A的位置包括x与y
    :param position_pixel_B: 所要计算的像素B的位置包括x与y
    :return: 计算的距离，类型int
    '''

    x_a, y_a = position_pixel_A  # 像素A在图片上的位置x与y
    x_b, y_b = position_pixel_B

    distance = (x_a - x_b) ** 2 + (y_a - y_b) ** 2
    distance = distance ** 0.5

    return int(distance)

def random_list(list, rstate):
    p = np.random.RandomState(seed=rstate).permutation(len(list))
    list = np.array(list)

    return list[p].tolist()


class OA_AA_Kappa():
    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    @staticmethod
    def AA_andEachClassAccuracy(confusion_matrix):
        counter = confusion_matrix.shape[0]
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

    @staticmethod
    def reports(y_pred, y_test, list_key=False):

        if list_key is True:
            y_pred = stretch_list(y_pred)
            y_test = stretch_list(y_test)

        # y_pred = y_pred.reshape(-1)
        classification = classification_report(y_test, y_pred)
        oa = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        each_acc, aa = OA_AA_Kappa.AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(y_test, y_pred)
        print("OA={:.5}, AA={:.5}, Kappa={:.5}".format(oa * 100, aa * 100, kappa * 100))
        print('each_class', end=':')
        for i, class_acc in enumerate(each_acc):
            print("{:.5}".format(class_acc), end=' ,')

        print('')
        predict_count = Counter(y_pred)
        groundTruth_count = Counter(y_test)
        predict_count = sorted(predict_count.items(), key=lambda x: x[0], reverse=False)
        groundTruth_count = sorted(groundTruth_count.items(), key=lambda x: x[0], reverse=False)
        print('predict_count:     {}'.format(predict_count))
        print('groundTruth_count: {}'.format(groundTruth_count))

        return oa * 100, aa * 100, kappa * 100, each_acc * 100
        # return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))
