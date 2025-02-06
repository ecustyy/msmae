import torch
import torch.nn as nn

import torchvision.transforms as trans
from torch.utils.data import Dataset
from copy import deepcopy
from lib import oct_lib
import warnings
warnings.filterwarnings('ignore')
from layers.vitcot import ViT
from lib.oct_dataloader import DataLoader
from lib import oct_lib, readData
import argparse, json
from lib.oct_lib import DModel, image_transform,image_read3d,slic3d
from lib.oct_lib import Losstry
from tensorboardX import SummaryWriter
import numpy as np
from layers.cnn3dsep import Net3d
import os

# 为当前阶段新建文件夹
folder_name = "save/oct_mask"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

# Config
with open('GraphTransformer_setting.json') as f:
    config = json.load(f)
data_path = config['data_path']
modality_filename = config['modality_filename']
saveroot = config['saveroot']
batchsize = config['oct_set']['batchsize']
lr = config['oct_set']['lr']
n_epochs = config['oct_set']['n_epochs']

#tensorboard
logger = SummaryWriter('output')  # tensorboard初始化一个写入单元

# 读取所有数据
data2d, datasetlist, labs, clist, Bclist, idn = readData.read_datasetme(data_path, modality_filename, saveroot)  # 读取文件路径和诊断真值
allBs = datasetlist[()]['OCT'] # ndarray转化为内置字典类型dict

#训练和评估双模态图像变换
img_transforms = trans.Compose([
    trans.ToTensor(),
])

# 设置训练数据加载器
train_loader = DataLoader(
    dataset=[allBs, labs[:-10], idn],
    img1_trans=img_transforms,
    img2_trans=img_transforms,
    batch_size=batchsize,
    num_workers=0
)
val_loader = DataLoader(
    dataset=[allBs, labs, idn],
    img1_trans=img_transforms,
    img2_trans=img_transforms,
    batch_size=batchsize,
    num_workers=0,
    spmod='seq'
)
#定义模型 优化器 准则
mmodel = Net3d()
# pretrained = r'F:\task9\me\ViT-OCT6.2\output\best_model_0.0007.pth'
# state_dict = torch.load(pretrained, map_location="cpu")
# msg = mmodel.load_state_dict(state_dict, strict=False)
# print("=> loaded pre-trained model '{}'".format(pretrained))
gmodel = ViT(
    num_patches = 125,
    num_classes = 7,
    dim = 256,
    depth = 6,
    heads = 16,
    last_heads = 64,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)
# model = baidu_lib.Model()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
model = DModel(mmodel,gmodel)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = Losstry() #默认为求均值

#训练
#best_model = baidu_lib.train_ID(model, train_loader, val_loader, criterion, optimizer, n_epochs, device)
oct_lib.train_ID(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, folder_name)