import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Net3d(nn.Module):
    def __init__(self):
        super(Net3d, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv3d(1, 2, 3, padding=1)
        self.conv2 = nn.Conv3d(2, 4, 3, padding=1)
        self.conv3 = nn.Conv3d(4, 8, 3, padding=1)
        self.conv4 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv5 = nn.Conv3d(16, 32, 3, padding=1)

        # 定义池化层
        self.pool2 = nn.MaxPool3d(2, 2)
        self.pool5 = nn.MaxPool3d(5, 5)

        # 定义全连接层
        self.fc1 = nn.Linear(32 * 125, 512)   #16个卷积核64个超像素（每个维度保留4份）
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        # 将输入数据增加一个通道维度
        x = x.unsqueeze(1)

        # 前向传播

        # 卷积层
        x = F.relu(self.conv1(x))   #[4,2,400,400,400]
        x = self.pool5(x)   #[4,2,80,80,80]
        x = F.relu(self.conv2(x))   #[4,4,80,80,80]
        x = self.pool2(x)   #[4,4,40,40,40]
        x = F.relu(self.conv3(x))   #[4,8,40,40,40]
        x = self.pool2(x)   #[4,8,20,20,20]
        x = F.relu(self.conv4(x))   #[4,16,20,20,20]
        x = self.pool2(x)   #[4,16,10,10,10]
        x = F.relu(self.conv5(x))   #[4,32,10,10,10]
        v = rearrange(x, 'x n (a b) (c d) (e f)-> x n a b c d e f', a=5,c=5,e=5)
        xv = rearrange(v, 'x n a b c d e f->x (a c e) (n b d f)', a=5, c=5, e=5)
        x = self.pool2(x)   #[4,16,5,5,5]

        # 全连接层
        x = x.view(-1, 32 * 125)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x,xv