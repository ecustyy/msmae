
import torch
import torch.nn as nn
import torchvision.transforms as trans
import warnings
warnings.filterwarnings('ignore')

from layers.vitcot import ViT
from lib import oct_lib, readData
from lib.oct_dataloader import DataLoader
from lib.oct_lib import test_me, test_oct_mask2
from lib.oct_lib import DModel
from layers.cnn3dsep import Net3d
import json
import os

# Config
with open('GraphTransformer_setting.json') as f:
    config = json.load(f)
data_path = config['data_path']
modality_filename = config['modality_filename']
saveroot = config['saveroot']
batchsize = config['oct_set']['batchsize']

# 读取所有数据
data2d, datasetlist, labs, clist, Bclist, idn = readData.read_datasetme(data_path, modality_filename, saveroot)  # 读取文件路径和诊断真值
allBs = datasetlist[()]['OCT'] # ndarray转化为内置字典类型dict

#测试程序双模态变换
img_transforms = trans.Compose([
    trans.ToTensor(),
])

val_loader = DataLoader(
    dataset=[allBs, labs, idn],
    img1_trans=img_transforms,
    img2_trans=img_transforms,
    batch_size=batchsize,
    num_workers=0,
    spmod='seq'
)

# model加载模型和权值
use_cuda = torch.cuda.is_available()
state_dict = torch.load(r'F:\task9\me\ViT-OCT6.6\output\best_model_0.0018.pth', map_location=torch.device('cpu'))
mmodel = Net3d()
gmodel = ViT(
    num_patches = 125,
    num_classes = 7,
    dim = 256,
    depth = 6,
    heads = 16,
    last_heads = 64,
    mlp_dim = 1024,
)
model = DModel(mmodel,gmodel)
model.load_state_dict(state_dict)
model.eval()

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

classes = ['Normal', 'CNV', 'DR', 'AMD', 'CSC', 'RVO', 'Others']  # 前四个是3M中使用的，
# model加载模型和权值
files= os.listdir('save/oct_mask/') #得到文件夹下的所有文件名称
for mdname in files: #遍历文件夹
    if mdname.endswith('.pth'):  # 如果检测到模型
        mdid = mdname.split('_')[2][:-4]
        path_experiment='save/oct_mask/'+mdid

        state_dict = torch.load('save/oct_mask/'+mdname, map_location=torch.device("cuda:0"))

        model.load_state_dict(state_dict)
        model.eval()

        if not os.path.exists(path_experiment):
            os.makedirs(path_experiment)
            print(f"Folder '{path_experiment}' created.")
        else:
            print(f"Folder '{path_experiment}' already exists.")

        test_oct_mask2(model, val_loader, savename=path_experiment)  # 计算指标不输出中间图像

print("done")
