import torch
import torch.nn as nn

import torchvision.transforms as trans
from torch.utils.data import Dataset
from copy import deepcopy
import baidu_lib
import warnings
warnings.filterwarnings('ignore')
from vitme import ViT
from mydataloader import DataLoader
import mydataloader2
import readData
import argparse, json
from graph_transformer_net import GraphTransformerNet
from baidu_lib import DModel
from losses import Losstry
from tensorboardX import SummaryWriter

# Config
batchsize = 28 # 4 patients per iter, i.e, 20 steps / epoch
oct_img_size = [512, 512]
image_size = 400
iters = 100 # For demonstration purposes only, far from reaching convergence
n_epochs = 150
val_ratio = 0.2 # 80 / 20
#testset_root = "val_data/multi-modality_images"
num_workers = 0
init_lr = 1e-4
optimizer_type = "adam"
use_cuda = True
data_path = 'E:\\task8\\OCT500\\OCTA-600\\'
modality_filename = ['OCT','FULL']
saveroot = 'D:/task8/logs'

#tensorboard
logger = SummaryWriter('output')  # tensorboard初始化一个写入单元

with open('GraphTransformer_setting.json') as f:
    config = json.load(f)
net_params = config['net_params']
net_params['device'] = torch.device("cuda")
net_params['gpu_id'] = 0
net_params['batch_size'] = 128

net_params['num_atom_type'] = 28
net_params['num_bond_type'] = 4
gmodel = GraphTransformerNet(net_params)


# 读取所有数据
data2d, datasetlist, labs, clist, Bclist, idn = readData.read_datasetme(data_path, modality_filename, saveroot)  # 读取文件路径和诊断真值

#训练和评估双模态图像变换
img_train_transforms = trans.Compose([
    trans.ToTensor(),
    trans.RandomResizedCrop(    #将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小
        image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    #trans.RandomRotation(30)
])

oct_train_transforms = trans.Compose([
    trans.ToTensor(),
    trans.CenterCrop(oct_img_size),
    trans.RandomHorizontalFlip(),
    #trans.RandomVerticalFlip()
])

img_val_transforms = trans.Compose([
    trans.ToTensor(),
    #trans.CenterCrop(image_size*2),
])

oct_val_transforms = trans.Compose([
    trans.ToTensor(),
    trans.CenterCrop(oct_img_size)
])

#示例图像加载
# _train = baidu_lib.GAMMA_sub1_dataset(dataset_root=trainset_root,
#                         img_transforms=img_train_transforms,
#                         oct_transforms=oct_train_transforms,
#                         label_file=gt_file)
#
# _val = baidu_lib.GAMMA_sub1_dataset(dataset_root=trainset_root,
#                         img_transforms=img_val_transforms,
#                         oct_transforms=oct_val_transforms,
#                         label_file=gt_file)

# 设置训练数据加载器
train_loader = DataLoader(
    dataset=[data2d, labs, clist, idn],
    img1_trans=img_train_transforms,
    img2_trans=oct_train_transforms,
    batch_size=batchsize,
    num_workers=0
)
val_loader = mydataloader2.DataLoader(
    dataset=[data2d, labs, clist, idn],
    img1_trans=img_val_transforms,
    img2_trans=oct_val_transforms,
    batch_size=30,
    num_workers=0
)
#定义模型 优化器 准则
mmodel = ViT(
    image_size = image_size,
    patch_size = 25,
    num_classes = 7,
    dim = 512,
    depth = 6,
    heads = 16,
    last_heads = 64,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)
# model = baidu_lib.Model()
device = torch.device("cuda:0" if use_cuda else "cpu")
model = mmodel
# model = DModel(mmodel,gmodel)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
criterion = Losstry()

#训练
#best_model = baidu_lib.train_ID(model, train_loader, val_loader, criterion, optimizer, n_epochs, device)
baidu_lib.train_ID(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, logger)