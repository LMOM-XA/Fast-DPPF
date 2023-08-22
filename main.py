import gc
import os
import time
import numpy as np
import torch

# 自己写的包
from config import DefaultConfig
import data.dataProcess as dataProcess
import utile.generalUtile as generalUtile
from data.HsiData import HsiData
from models import SSCNNmodule
from draw.draw_pics import draw_np_pic

opt = DefaultConfig


# 加载model、dataLoader
def initial(model_name, init_class_num, model_load_path=''):
    """
    :param model_name:要加载的模型名称
    :param init_class_num: 定义网络分类层神经元数目
    :param model_load_path:若测试或验证是，需要加载训练好的模型，这是模型的路径
    :return:
    """
    # 加载模型
    data_name = opt.data_name
    if model_name == 'PPF':
        init_model = SSCNNmodule.SSCNN(data_name=data_name, num_class=init_class_num, model_name='PPF')
    elif model_name == 'ODPPF':
        init_model = SSCNNmodule.SSCNN(data_name=data_name, num_class=init_class_num, model_name='ODPPF')
    else:
        init_model = 0

    init_model.to(opt.device)  # 上cpu或gpu

    # step3 优化器
    init_criterion = torch.nn.CrossEntropyLoss()
    init_optimizer = torch.optim.SGD(init_model.parameters(), opt.lr,
                                     momentum=opt.momentum,
                                     weight_decay=opt.weight_decay)
    # init_optimizer = torch.optim.Adam(init_model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # step4 GPU一些调试
    if opt.use_gpu:
        torch.backends.cudnn.enabled = True  # cuDNN使用非确定性算法
        torch.backends.cudnn.benchmark = True  # 网络的输入数据维度或类型上变化不大 可以增加运行效率

    # step5 载入之前的训练
    # pre_epoch = 0
    if model_load_path is not '':
        if opt.use_gpu:
            checkpoint = torch.load(model_load_path)
        else:
            checkpoint = torch.load(model_load_path, map_location=torch.device('cpu'))
        init_model.load_state_dict(checkpoint['net'])  # state_dictvg
        init_optimizer.load_state_dict(checkpoint['optimizer'])
        # pre_epoch = checkpoint['epoch']

    return init_model, init_criterion, init_optimizer


# 一个epoch中的训练
def train_one_epoch(train_loader_func, train_model, train_criterion, train_optimizer):
    train_model.train()

    train_num = np.zeros(len(train_loader_func))
    train_correct = np.zeros(len(train_loader_func))
    train_loss = np.zeros(len(train_loader_func))

    tttt = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader_func):
        # 网络过程
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)
        outputs = train_model(inputs, True)
        loss = train_criterion(outputs, labels)
        loss.backward()
        train_optimizer.step()
        train_optimizer.zero_grad()

        # loss,acc的计算
        batch_loss, batch_correct = generalUtile.train_acc_loss(loss, outputs, labels)
        train_num[batch_idx] = labels.size(0)
        train_loss[batch_idx] = batch_loss
        train_correct[batch_idx] = batch_correct

    train_loss_epoch = sum(train_loss)
    train_correct_epoch = sum(train_correct)
    train_num_epoch = sum(train_num)
    return train_loss_epoch, train_correct_epoch, train_num_epoch


def train(train_loader_func, train_model, train_criterion, train_optimizer, epoch, val_loader_func):
    best_acc = 0
    for e in range(epoch):
        # generalUtile.adjust_learning_rate(optimizer, e)
        start_time = time.time()  # 记录时间
        loss, correct, num = train_one_epoch(train_loader_func, train_model, train_criterion, train_optimizer)

        # print("trainLoss = {}, trainTotal = {}, TrainCorrect = {}".format(loss, num, correct))
        print('train {} epoch loss: {}   acc: {:.5}  spend time {:.5}'.format(e, loss / num, 100 * correct / num,
                                                                              time.time() - start_time))

        if opt.validation is True and e % 3 == 0 and e != 0:
            val_acc = validation(val_loader_func, train_model)
            if val_acc > best_acc and val_acc > 0.65:
                best_acc = val_acc
                train_model.save(train_optimizer, e, best_acc)


# 网络测试
def inference(model_name, test_class_num, test_loader_func, model_load_path):
    test_model, test_criterion, test_optimizer = initial(model_name, test_class_num, model_load_path)
    test_model.eval()
    with torch.no_grad():
        model_predict = []  # 之后用来计算OA_AA_Kappa
        ground_truth = []  # 同上

        # 预测过程
        for batch_idx, (inputs, labels) in enumerate(test_loader_func):

            i_shape = inputs.shape
            i_band = i_shape[-1]
            i_batch_size = i_shape[0]

            inputs = inputs.reshape(-1, 2, i_band)  # 解释在validation()函数中
            # 网络过程
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)
            outputs = test_model(inputs, False)  # 这里的False是因为训练集的输出与测试集的输出不一样，测试集中要去掉0类的输出

            outputs = outputs.to('cpu')  # 其余操作在CPU上进行其他运算要快一些（经过测试，CPU是上24s，GPU上37s）
            batch_output = torch.chunk(outputs, i_batch_size, 0)  # 切割为batch个块,
            # 每个batch 进行决策
            predict_label_batch = torch.zeros([i_batch_size], dtype=torch.int) - 1  # 每个batch的预测结果

            for i_batch, every_output in enumerate(batch_output):

                _, train_predicted = torch.max(every_output.data, 1)  # (1是每行的最大值)
                train_predicted = train_predicted + 1  # 在测试网络中去掉了label为0的输出神经元，防止数据错位，得+1
                # 取众数
                predict_label = torch.mode(train_predicted)[0]  # 取出投票值
                predict_label_batch[i_batch] = predict_label
                if predict_label == 0:
                    print(0)
            model_predict.append(predict_label_batch.numpy())
            ground_truth.append(labels.data.cpu().numpy())

        accuracy = generalUtile.OA_AA_Kappa.reports(model_predict, ground_truth, list_key=True)  # 函数里面包含了print
        return accuracy


def validation(val_loader_func, val_model):
    start_time = time.time()  # 记录时间
    val_model.eval()
    with torch.no_grad():
        val_num = np.zeros(len(val_loader_func))
        val_correct = np.zeros(len(val_loader_func))

        for batch_idx, (inputs, labels) in enumerate(val_loader_func):
            # 网络过程
            i_shape = inputs.shape
            i_band = i_shape[-1]
            i_batch_size = i_shape[0]

            # 网络输入input——>[batch,25,1,2]
            # 总共有batch个input，每个input是（25*1*2），这里的24是邻域大小
            # 首先把它拉长为(batch*25,1,2,200) 丢入网络（格式要求），之后在切割为batch个块，每个batch需要有自己的决策。
            inputs = inputs.reshape(-1, 2, i_band)

            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)
            outputs = val_model(inputs, False)  # 这里的False是因为训练集的输出与测试集的输出不一样，测试集中要去掉0类的输出

            outputs = outputs.to('cpu')  # 其余操作在CPU上进行其他运算要快一些（经过测试，CPU是上24s，GPU上37s）
            batch_output = torch.chunk(outputs, i_batch_size, 0)  # 切割为batch个块,
            # 每个batch 进行决策
            predict_label_batch = torch.zeros([i_batch_size], dtype=torch.int) - 1  # 每个batch的预测结果
            for i_batch, every_output in enumerate(batch_output):
                # 遇到用0填充的数据的问题
                _, train_predicted = torch.max(every_output.data, 1)  # 这里可能要改(1是每行的最大值)
                train_predicted = train_predicted + 1  # 在测试网络中去掉了label为0的输出神经元，防止数据错位，得+1
                # 取众数
                predict_label = torch.mode(train_predicted)[0]  # 取出投票值
                predict_label_batch[i_batch] = predict_label

            val_correct[batch_idx] = torch.sum(predict_label_batch == labels.data.cpu())
            val_num[batch_idx] = labels.size(0)

            # # loss,acc的计算
            # _, batch_correct = generalUtile.train_acc_loss(torch.tensor(0), outputs, labels)
            # val_num[batch_idx] = labels.size(0)
            # # total_loss[batch_idx] = batch_loss
            # val_correct[batch_idx] = batch_correct

        val_correct_epoch = sum(val_correct)
        val_num_epoch = sum(val_num)
        acc = val_correct_epoch / val_num_epoch
        print('val acc={:.5} spend time {:.5}'.format(acc * 100, time.time() - start_time))
        return acc


# 作图
def inference_picture(model_name, test_class_num, data_shaper, test_loader_func, model_load_path):
    start_time = time.time()  # 记录时间
    model, _, _ = initial(model_name, test_class_num, model_load_path)
    model.eval()
    with torch.no_grad():
        predict_pic = np.zeros((data_shaper[0] * data_shaper[1]), dtype='int32')
        count = 0

        # 预测过程
        for batch_idx, (inputs, labels) in enumerate(test_loader_func):

            i_shape = inputs.shape
            i_band = i_shape[-1]
            i_batch_size = i_shape[0]

            inputs = inputs.reshape(-1, 2, i_band)  # 解释在validation()函数中
            # 网络过程
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)
            outputs = model(inputs, False)  # 这里的False是因为训练集的输出与测试集的输出不一样，测试集中要去掉0类的输出

            outputs = outputs.to('cpu')  # 其余操作在CPU上进行其他运算要快一些（经过测试，CPU是上24s，GPU上37s）
            batch_output = torch.chunk(outputs, i_batch_size, 0)  # 切割为batch个块,
            # 每个batch 进行决策
            predict_label_batch = torch.zeros([i_batch_size], dtype=torch.int) - 1  # 每个batch的预测结果

            for i_batch, every_output in enumerate(batch_output):

                _, train_predicted = torch.max(every_output.data, 1)  # (1是每行的最大值)
                train_predicted = train_predicted + 1  # 在测试网络中去掉了label为0的输出神经元，防止数据错位，得+1
                # 取众数
                predict_label = torch.mode(train_predicted)[0]  # 取出投票值
                predict_label_batch[i_batch] = predict_label
                if predict_label == 0:
                    print(0)

            for res in predict_label_batch:
                predict_pic[count] = res
                count += 1

        predict_pic = predict_pic.reshape(data_shaper[0], data_shaper[1])
        print('测试用时为：{:.5}'.format(time.time() - start_time))
        return predict_pic


def train_total():
    # dataset = ['IP', 'Sa', 'UP', 'Pavia', 'KSC']
    # dataset = ['IP', 'Sa', 'UP', 'Pavia']
    # dataset = ['KSC']
    dataset = ['UP']
    # model_names = ['PPF', 'ODPPF']
    model_names = ['ODPPF']
    # random_num = [111, 7288, 5295, 2502, 5075, 165, 6710, 5238, 9700, 4040, 5771, 231, 45156, 2412, 802]
    random_num = [111, 7288, 5295, 2502, 5075]

    for model_name in model_names:
        opt.model = model_name
        print("#################"
              "#########################################")
        print("model={}".format(model_name))
        print("##########################################################")

        for name in dataset:
            opt.data_name = name

            for i in random_num:
                opt.rand_state = i
                patch_size = opt.patch_size
                batch_size = opt.batch_size
                # 测试验证训练集定义
                disjointed = opt.disjointed  # 是否为disjointed划分，FALSE是随机划分
                number_threshold = opt.number_threshold
                hsi = HsiData(name, model_name, opt.train_set_proportion, disjointed=disjointed,
                              number_threshold=number_threshold)

                train_loader = dataProcess.DataLoader.getin_dataloader_train(hsi.origin_data, hsi.train_set_paired,
                                                                             hsi.train_label_paired,
                                                                             shuffle=True, batch_size=batch_size)
                test_loader = dataProcess.DataLoader.getin_dataloader_test(hsi.origin_data, hsi.origin_label,
                                                                           hsi.test_set, patch_size=patch_size,
                                                                           shuffle=False, batch_size=batch_size * 10)
                class_num = len(np.unique(hsi.train_label_paired))

                # 模型 优化器定义
                epoch = opt.epoch
                # data_shaper = hsi.hsi_shape
                model, criterion, optimizer = initial(model_name, class_num)
                ##############
                train(train_loader, model, criterion, optimizer, epoch, test_loader)
                #############

                # 回收掉占内存的东西，没钱买内存，没办法啊
                del hsi, train_loader, test_loader, model
                gc.collect()


def test_total():
    # paths = ['saveParameter/random', 'saveParameter/block','saveParameter/bottom']
    # paths = ['saveParameter']
    paths = ['saveParameter/temp']
    for path in paths:
        for file_name in os.listdir(path):

            if file_name.endswith('.save'):

                print("##########################################################")
                print("file={}".format(file_name))
                print("##########################################################")

                model_load_path = os.path.join(path, file_name)
                model_name, seed, data_name, _ = file_name.split('_')
                seed = int(seed[4:])

                opt.model = model_name
                opt.data_name = data_name
                opt.rand_state = seed
                patch_size = opt.patch_size
                batch_size = opt.batch_size
                disjointed = opt.disjointed
                number_threshold = opt.number_threshold

                if model_name in ['PPF', 'ODPPF']:
                    opt.patch_size = 5
                else:
                    print("model_name不正确程序退出")
                    exit(0)

                hsi = HsiData(data_name, model_name, opt.train_set_proportion, disjointed=disjointed,
                              number_threshold=number_threshold)

                test_loader = dataProcess.DataLoader.getin_dataloader_test(hsi.origin_data, hsi.origin_label,
                                                                           hsi.test_set, patch_size=patch_size,
                                                                           shuffle=False, batch_size=batch_size * 10)
                class_num = len(np.unique(hsi.train_label_paired))

                accuracy = inference(model_name, class_num, test_loader, model_load_path)

                # 下面是输出到文件
                oa, aa, kappa, each_class = accuracy
                EAC = ''  # 每个类别的精度
                for each in each_class:
                    temp = '{:.2f},'.format(each)
                    EAC += temp
                fileName = os.path.join(path, 'test.txt')
                line = '{},{},{},{:.2f},{:.2f},{:.2f},each,{}-1\n'.format(data_name, seed, model_name, oa, aa, kappa,
                                                                          EAC)
                with open(fileName, 'a', encoding='utf-8') as file:

                    file.write(line)


def draw_pic_total():
    # paths = ['saveParameter/draw/random', 'saveParameter/draw/block', 'saveParameter/draw/bottom']
    paths = ['saveParameter/temp']
    for path in paths:
        for file_name in os.listdir(path):

            if file_name.endswith('.save'):

                print("##########################################################")
                print("file={}".format(file_name))
                print("##########################################################")

                model_load_path = os.path.join(path, file_name)
                model_name, seed, data_name, _ = file_name.split('_')
                seed = int(seed[4:])

                opt.model = model_name
                opt.data_name = data_name
                opt.rand_state = seed
                patch_size = opt.patch_size
                batch_size = opt.batch_size
                disjointed = opt.disjointed
                number_threshold = opt.number_threshold

                if model_name in ['PPF', 'ODPPF']:
                    opt.patch_size = 5
                else:
                    print("model_name不正确程序退出")
                    exit(0)

                hsi = HsiData(data_name, model_name, opt.train_set_proportion, disjointed=disjointed,
                              number_threshold=number_threshold)

                test_loader = dataProcess.DataLoader.get_pic_dataloader(hsi.origin_data, hsi.origin_label,
                                                                        hsi.hsi_shape, patch_size=patch_size,
                                                                        shuffle=False, batch_size=batch_size * 10)
                class_num = len(np.unique(hsi.train_label_paired))
                predict_pic = inference_picture(model_name, class_num, hsi.hsi_shape, test_loader, model_load_path)

                # 下面是作图并保存
                # _, _, disjointed = path.split('/')
                _, disjointed = path.split('/')
                save_root = "saveParameter/draw/{}_{}_{}.jpg".format(data_name, model_name, disjointed)
                draw_np_pic(predict_pic, save_root)
                print("图片保存至:{}".format(save_root))


if __name__ == '__main__':
    # train_total()
    # test_total()
    draw_pic_total()
    pass
