import torch
import torchvision.transforms as trans
from lib import cfp_lib, readData
import warnings
warnings.filterwarnings('ignore')
from layers.vitme import ViT
from lib import cfp_dataloader,cfp_dataloaderbl
import argparse, json
import os

# 为当前阶段新建文件夹
folder_name = "save/cfp_mask"
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
batchsize = config['cfp_set']['batchsize']
lr = config['cfp_set']['lr']
n_epochs = config['cfp_set']['n_epochs']
image_size = config['cfp_set']['image_size']

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

img_val_transforms = trans.Compose([
    trans.ToTensor(),
    #trans.CenterCrop(image_size*2),
])

# 设置训练数据加载器
train_loader = cfp_dataloaderbl.DataLoader(
    dataset=[data2d, labs, clist, idn],
    img1_trans=img_train_transforms,
    batch_size=batchsize,
    num_workers=0
)
val_loader = cfp_dataloader.DataLoader(
    dataset=[data2d, labs, clist, idn],
    img1_trans=img_val_transforms,
    batch_size=30,
    num_workers=0,
    spmod='seq'
)
#定义模型 优化器 准则
model = ViT(
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
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# model = DModel(mmodel,gmodel)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = cfp_lib.Losstry()

#训练
#best_model = baidu_lib.train_ID(model, train_loader, val_loader, criterion, optimizer, n_epochs, device)
cfp_lib.train_ID(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, folder_name)