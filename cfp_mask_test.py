import os
import torch
import torch.nn as nn
import argparse, json
import torchvision.transforms as trans
import warnings
warnings.filterwarnings('ignore')

from layers.vitme import ViT
from lib import readData
from lib.cfp_dataloader import DataLoader
from lib.cfp_lib import test_cfp_mask, test_cfp_mask2

# Config
with open('GraphTransformer_setting.json') as f:
    config = json.load(f)

data_path = config['data_path']
modality_filename = config['modality_filename']
saveroot = config['saveroot']
batchsize = config['cfp_set']['batchsize']
lr = config['cfp_set']['lr']
n_epochs = config['cfp_set']['n_epochs']
image_size = config['cfp_set']['image_size']

# 读取所有数据
data2d, datasetlist, labs, clist, Bclist, idn = readData.read_datasetme(data_path, modality_filename, saveroot)  # 读取文件路径和诊断真值

#测试程序双模态变换
img_val_transforms = trans.Compose([
    trans.ToTensor(),
    #trans.CenterCrop(image_size*2),
])

val_loader = DataLoader(
    dataset=[data2d, labs, clist, idn],
    img1_trans=img_val_transforms,
    batch_size=batchsize,
    num_workers=0,
    spmod = 'seq'
)

model = ViT(
    image_size = image_size,
    patch_size = 25,
    num_classes = 7,
    dim = 512,
    depth = 6,
    heads = 16,
    last_heads = 64,
    mlp_dim = 1024
)
model.cuda()

classes = ['Normal', 'CNV', 'DR', 'AMD', 'CSC', 'RVO', 'Others']  # 前四个是3M中使用的，
# model加载模型和权值
files= os.listdir('save/cfp_mask/') #得到文件夹下的所有文件名称

for mdname in files: #遍历文件夹
    if mdname.endswith('.pth'):  # 如果检测到模型
        mdid = mdname.split('_')[2][:-4]
        path_experiment='save/cfp_mask/'+mdid

        state_dict = torch.load('save/cfp_mask/'+mdname, map_location=torch.device("cuda:0"))

        model.load_state_dict(state_dict)
        model.eval()

        if not os.path.exists(path_experiment):
            os.makedirs(path_experiment)
            print(f"Folder '{path_experiment}' created.")
        else:
            print(f"Folder '{path_experiment}' already exists.")

        # 计算指标
        # test_cfp_mask(model, val_loader, savename=path_experiment)  #输出中间图像
        test_cfp_mask2(model, val_loader, savename=path_experiment)  #计算指标不输出中间图像
print("done")
