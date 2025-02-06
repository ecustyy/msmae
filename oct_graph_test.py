import torch
import torch.nn as nn

import torchvision.transforms as trans
from lib import graph_lib
import warnings
warnings.filterwarnings('ignore')
from lib.graph_loader import DataLoader
import json
from layers.graph_transformer_net import GraphTransformerNet
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as trans
import warnings
warnings.filterwarnings('ignore')
from lib.graph_lib import test_me
import os

# Config
with open('GraphTransformer_setting.json') as f:
    config = json.load(f)
saveroot = r"F:\task9\me\ViT-OCT6.6\output\all-data.npz"
batchsize = 3

#定义模型
net_params = config['oct_params']
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

# 设置测试数据加载器
val_loader = DataLoader(
    dataset=[labels, xalls, attalls],
    img1_trans=img_transforms,
    img2_trans=img_transforms,
    batch_size=batchsize,
    num_workers=0,
    spmod='seq'
)

# model加载模型和权值
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# model = DModel(mmodel,gmodel)
model.to(device)

classes = ['Normal', 'CNV', 'DR', 'AMD', 'CSC', 'RVO', 'Others']  # 前四个是3M中使用的，
# model加载模型和权值
filedir='save/oct_graph/'
files= os.listdir(filedir) #得到文件夹下的所有文件名称

for mdname in files: #遍历文件夹
    if mdname.endswith('.pth'):  # 如果检测到模型
        mdid = mdname.split('_')[2][:-4]
        path_experiment=filedir+mdid

        state_dict = torch.load(filedir+mdname, map_location=torch.device("cuda:0"))

        model.load_state_dict(state_dict)
        model.eval()

        if not os.path.exists(path_experiment):
            os.makedirs(path_experiment)
            print(f"Folder '{path_experiment}' created.")
        else:
            print(f"Folder '{path_experiment}' already exists.")

        # 计算指标
        test_me(model, val_loader, savename=path_experiment)

print("done")
