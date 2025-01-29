
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34


#model = EfficientNet.from_name(args.arch)
#model = EfficientNet.from_pretrained('efficientnet-b0')
#from torchvision import transforms, datasets
import torchvision.transforms as trans
from torch.utils.data import Dataset
import baidu_lib
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

from vitme import ViT
import readData
from mydataloader2 import DataLoader
from test_lib import test_me

# Config
batchsize = 10 # 4 patients per iter, i.e, 20 steps / epoch
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

# 读取所有数据
data2d, datasetlist, labs, clist, Bclist, idn = readData.read_datasetme(data_path, modality_filename, saveroot)  # 读取文件路径和诊断真值

#测试程序双模态变换
img_val_transforms = trans.Compose([
    trans.ToTensor(),
    #trans.CenterCrop(image_size*2),
])

oct_val_transforms = trans.Compose([
    trans.ToTensor(),
    trans.CenterCrop(oct_img_size)
])

val_loader = DataLoader(
    dataset=[data2d, labs, clist, idn],
    img1_trans=img_val_transforms,
    img2_trans=oct_val_transforms,
    batch_size=batchsize,
    num_workers=0,
    spmod = 'seq'
)

# model加载模型和权值
use_cuda = torch.cuda.is_available()
state_dict = torch.load('output/best_model_0.8189.pth', map_location=torch.device('cpu'))
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
model.load_state_dict(state_dict)
model.eval()

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

criterion = nn.CrossEntropyLoss()

classes = ['Normal', 'CNV', 'DR', 'AMD', 'CSC', 'RVO', 'Others']  # 前四个是3M中使用的，
# 计算指标
test_me(model, val_loader, classes)

print("done")
