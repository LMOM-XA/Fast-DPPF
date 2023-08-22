import torch


class DefaultConfig:
    # HSI名称
    data_name = 'IP'
    # 原始数据路径、标记存放路径
    origin_data_root = {
        'IP': ['data/originData/Indian_pines_corrected.mat', 'indian_pines_corrected'],
        'UP': ['data/originData/PaviaU.mat', 'paviaU'],
        'Sa': ['data/originData/Salinas_corrected.mat', 'salinas_corrected'],
        'KSC': ['data/originData/KSC.mat', 'KSC'],
        'Pavia': ['data/originData/Pavia.mat', 'pavia'],
        'Botswana': ['data/originData/Botswana.mat', 'Botswana']
    }

    origin_label_data_root = {
        # 'IP': ['data/originData/Indian_pines_gt.mat', 'indian_pines_gt'],
        'IP': ['data/originData/Indian_pines_gt_LMOM.mat', 'indian_pines_gt'],
        'UP': ['data/originData/PaviaU_gt.mat', 'paviaU_gt'],
        'Sa': ['data/originData/Salinas_gt.mat', 'salinas_gt'],
        'KSC': ['data/originData/KSC_gt.mat', 'KSC_gt'],
        'Pavia': ['data/originData/Pavia_gt.mat', 'pavia_gt'],
        'Botswana': ['data/originData/Botswana_gt.mat', 'Botswana_gt']
    }

    data_shaper = {
        'IP': [145, 145, 200],
        'UP': [610, 340, 103],
        'Sa': [512, 217, 204],
        'KSC': [512, 614, 176],
        'Pavia': [1096, 715, 102],
        'Botswana': [1476, 256, 145]
    }

    patch_size = 5
    # 使用的模型，以及原始数据的patch
    # model = 'pResNet'  # 'pResNet', '3DCNN' 'PPF' 'ODPPF', '1DCNN', 'BS2T', 'SSRN', 'A2S2KResNet', 'RSSAN'
    # if model in ['pResNet', '3DCNN', 'BS2Tmodule', 'SSRN', 'A2S2KResNet', 'RSSAN', 'SSTN']:
    #     use_patch = True
    #     patch_size = 5
    # else:
    #     use_patch = False
    #     patch_size = 1

    # GPU调试
    use_gpu = False # (True if torch.cuda.is_available() else False)
    device = "cpu"  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理与装载
    rand_state = 53
    batch_size = 512
    # remove_redundant_IP = True  # 是否删掉IP数据集中那些数量少的样本
    train_set_proportion = 200  # 大于1时表示取固定个数，为0--1时表示取百分比训练集。()
    validation = True
    val_set_proportion = 0.1  # 验证集在测试集中占比
    disjointed = 'random'  # 'random'表示随机划分, 'block'表示按区块disjointed划分，'topBottom'表示从上至下每个类别选取去一半作为训练集。
    number_threshold = 50  # ODPPF中选取的配对数量

    # 优化器
    weight_decay = 1e-8
    momentum = 0.8
    # RSSAN 要用0.0003, 其余0.001
    lr = 0.001

    epoch = 100
    openMax = False  # 是否使用开放集分类（原始数据集的label是从1--numclass，但神经元是从0--numclass-1），使用开放集才是从0--numclass
    confidence_threshold = 0.7  # 伪标签置信度的阈值，达到这个阈值就可用于新一轮训练

    model_save_root = 'saveParameter'  # 保存模型路径
    model_load_root = 'saveParameter/SSRN_UP_200Best.save'  # 加载模型路径

