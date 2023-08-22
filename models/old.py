import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from config import DefaultConfig
import os

opt = DefaultConfig


class SSCNN(nn.Module):
    def __init__(self, data_name, num_class, model_name):
        super(SSCNN, self).__init__()

        self.data_name = data_name
        self.num_class = num_class
        self.model_name = model_name
        # self.train_test = train_or_test  # 控制是train还是test，test中不输出0类，比train中少一个类，True为Train

        self.cov_reduce_2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(2, 1)),
            nn.ReLU(True),
        )

        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(10, 10, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(10, 10, 3, 2),

            nn.ReLU(inplace=True)
        )

        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(10, 20, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(20, 20, 3, 2),

            nn.ReLU(inplace=True)
        )
        self.conv1d_3 = nn.Sequential(
            nn.Conv1d(20, 40, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(40, 40, 3, 2),

            nn.ReLU(inplace=True)
        )
        self.conv1d_4 = nn.Sequential(
            nn.Conv1d(40, 80, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(80, 80, 3, 2),

            nn.ReLU(inplace=True)
        )

        fist_linear = {
            'IP': 2540,
            'UP': 1100,
            'Sa': 2560,
            'KSC': 10,
            'Pavia': 1100,
            'Botswana': 10
        }

        fc = fist_linear[self.data_name]

        self.fc1 = nn.Linear(fc, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, self.num_class)

    # x shape = [100,1,200]
    def forward(self, x, train_test=True):  # [128,1,2,200]
        batch_size = x.shape[0]  # 记录batch的数量

        # 目前X是3维[128, 2, 200], 但2D卷积需要4维,需要扩展一维，扩展以后是[128, 1, 2, 200]
        x = torch.unsqueeze(x, 1)

        con_reduce = self.cov_reduce_2d(x)  # [128,10,1,200]
        con_reduce = con_reduce.reshape(batch_size, 10, -1)  # [128,10,200]

        conv1 = self.conv1d_1(con_reduce)  # [100,10,98]
        conv2 = self.conv1d_2(conv1)  # [100,20,47]
        conv3 = self.conv1d_3(conv2)  # [100,40,22]
        conv4 = self.conv1d_4(conv3)  # [100,80,9]

        # x = x.reshape(batch_size, -1)  # [100,400]
        con_reduce = con_reduce.reshape(batch_size, -1)  # [100,2000]
        conv1 = conv1.reshape(batch_size, -1)  # [100,980]
        conv2 = conv2.reshape(batch_size, -1)  # [100,940]
        conv3 = conv3.reshape(batch_size, -1)  # [100,880]
        conv4 = conv4.reshape(batch_size, -1)  # [100,720]

        conv_total = [conv4, conv3, conv2, conv1, con_reduce]
        out = torch.cat(conv_total[0: 3], dim=1)

        out = F.relu(self.fc1(out))  # [100,1024]
        out = F.relu(self.fc2(out))  # [100,256]
        out = F.relu(self.fc3(out))  # [100,64]
        out = (self.fc4(out))  # [100,17]

        if train_test:
            return out
        else:
            # 去除掉label为0的神经元输出,但在之后网络的输出中要+1，不然错位
            one_to_classes = range(1, self.num_class)
            out = torch.index_select(out, 1, torch.tensor(one_to_classes).to(opt.device))  # CUP的时候就把cuda删了
            return out

    def save(self, optimizer_state, epoch, loss):

        file_path = os.path.join(opt.model_save_root,
                                 "{}_seed{}_{}_{}Best.save".format(self.model_name, opt.rand_state, opt.data_name,
                                                                     opt.train_set_proportion))
        state = {'net': self.state_dict(), 'optimizer': optimizer_state.state_dict(), 'epoch': epoch}
        torch.save(state, file_path)
