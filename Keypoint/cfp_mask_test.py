import os
import torch
import models
import utils
import data
import random
import numpy as np
import cv2 as cv
from PIL import Image

from data import augmentation
from drawersmm import CAMDrawer
import torchvision.transforms as trans
import baidu_lib
from utils import test_dataloader
import cv2
import torchvision.utils as vutils
import sys
from cam import *

def show_cam(img, mask, title=None):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        img (Tensor): shape (1, 3, H, W)
        mask (Tensor): shape (1, 1, H, W)
    Return:
        heatmap (Tensor): shape (3, H, W)
        cam (Tensor): synthesized GradCAM cam of the same shape with heatmap.
        :param title:
    """
    mask = (mask - mask.min()).div(mask.max() - mask.min()).data
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().float()), cv2.COLORMAP_JET)  # [H, W, 3]
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # cam = heatmap + img.cpu()
    cam = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8) * heatmap
    # cam = (cam - cam.min()) / (cam.max() - cam.min())
    if title is not None:
        vutils.save_image(cam, title)

    return cam

sys.path.append(os.path.join(os.getcwd(), 'camconditioned_pix2pixHD'))
from camconditioned_pix2pixHD.util.util import load_cam, tensor2im
from camconditioned_pix2pixHD.models.models import create_model
from camconditioned_pix2pixHD.options.test_options import TestOptions

# Config
batchsize = 4 # 4 patients per iter, i.e, 20 steps / epoch
oct_img_size = [224, 224]
image_size = 224
iters = 100 # For demonstration purposes only, far from reaching convergence
n_epochs = 20
val_ratio = 0.2 # 80 / 20
trainset_root = "E:/task8/GAMMA/training_data/multi-modality_images"
gt_file = 'E:/task8/GAMMA/training_data/glaucoma_grading_training_GT.xlsx'
#testset_root = "val_data/multi-modality_images"
num_workers = 0
init_lr = 1e-4
optimizer_type = "adam"
use_cuda = True

#Step 1. Load a trained Multi-modal model
device = 0
device = torch.device("cuda:{}".format(device) if (torch.cuda.is_available() and device != "cpu") else "cpu")
configs = utils.load_config("config-cfp.py")
# configs.heatmap = True

label_list = '0 1 2'.split() # normal, dryAMD, pcv and wetAMD
n_classes = len(label_list)
checkpoint = "model/best_epoch51_0.6533.pth"

model = models.load_single_stream_model(configs, device, checkpoint)
# model.eval()
model.cuda()

gc = GradCAM(model, target_layer='layer3.1.bn2')

#Step 2. 读取多模态图像
#加载数据集
filelists = os.listdir(trainset_root)
print("Total Nums: {}".format(len(filelists)))

img_val_transforms = trans.Compose([
    trans.ToTensor(),
    # trans.CenterCrop(image_size*2),
    trans.Resize((image_size, image_size))
])

oct_val_transforms = trans.Compose([
    trans.ToTensor(),
    trans.CenterCrop(oct_img_size)
])

# 定义数据集和读取器
val_dataset = baidu_lib.GAMMA_sub1_dataset(dataset_root=trainset_root,
                                           img_transforms=img_val_transforms,
                                           filelists=filelists,
                                           label_file=gt_file)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    num_workers=num_workers
)

# for i, data in enumerate(val_loader, 0):  # 从0开始索引
#     # load batch
#     raw_im = data[0]
#     fundus_imgs = data[0].to(device=device, dtype=torch.float32)
#     label = data[1].to(device=device, dtype=torch.long)
#     # generate original CAMs
#     cams = cam_drawer.get_cam(fundus_imgs)
#     splicing = CAMDrawer.sequence_visualize(raw_im, cams, visual_size=(224,224),i=i)
savename='output/'
device = 'cuda:0'
pres = []
heatmaps = []
for i, data in enumerate(val_loader, 0):
    # load batch    data[3]为文件名
    fundus_img = data[0].to(device=device, dtype=torch.float32)    #[1,3,224,224]
    # fundus_img = fundus_imgs[1].unsqueeze(0)
    heatmap = gc(fundus_img , class_idx=None) # norm_image.cuda()   .cpu().data
    impath = savename + '/' + data[2][0] + '.png'
    cam = show_cam(fundus_img.cpu(), heatmap.cpu().data, impath)
    heatmaps.append(heatmap.cpu().numpy())

# 保存所有数据
np.savez(savename+'/cfp-mask-data', heatmaps)