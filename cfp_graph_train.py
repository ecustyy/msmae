import torch
import torch.nn as nn
import os
import torchvision.transforms as trans
from lib import graph_lib
import warnings
warnings.filterwarnings('ignore')
from lib.graph_loader import DataLoader
import json
from layers.graph_transformer_net import GraphTransformerNet
from tensorboardX import SummaryWriter
import numpy as np

# 为当前阶段新建文件夹
folder_name = "save/cfp_graph"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

# Config
with open('GraphTransformer_setting.json') as f:
    config = json.load(f)

saveroot = r"F:\task9\me\ViT-OCT3.3\output\all-data.npz"
batchsize = 3   #这是比较关键的值，可以确保训练的稳定
lr = 4e-6
n_epochs = config['cfp_set']['n_epochs']

net_params = config['cfp_params']
net_params['device'] = torch.device("cuda")
net_params['gpu_id'] = 0
net_params['batch_size'] = 128

net_params['num_atom_type'] = 28
net_params['num_bond_type'] = 4

model = GraphTransformerNet(net_params)

#tensorboard
logger = SummaryWriter('output')  # tensorboard初始化一个写入单元

# 读取保存的数据
f = np.load(saveroot, allow_pickle=True)
fnlist = f['arr_0']
pres = f['arr_1']
xalls = f['arr_2']
attalls = f['arr_3']
labels = f['arr_4']

img_transforms = trans.Compose([
    trans.ToTensor(),
])

# 设置训练数据加载器
train_loader = DataLoader(
    dataset=[labels, xalls, attalls],
    img1_trans=img_transforms,
    img2_trans=img_transforms,
    batch_size=batchsize,
    num_workers=0,
    spmod='rand'
)
val_loader = DataLoader(
    dataset=[labels, xalls, attalls],
    img1_trans=img_transforms,
    img2_trans=img_transforms,
    batch_size=batchsize,
    num_workers=0,
    spmod='seq'
)
#定义模型 优化器 准则

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# model = DModel(mmodel,gmodel)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss() #默认为求均值

#训练
#best_model = baidu_lib.train_ID(model, train_loader, val_loader, criterion, optimizer, n_epochs, device)
graph_lib.train_ID(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, folder_name)