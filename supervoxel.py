import os
import natsort
import numpy as np
import imageio
from mayavi import mlab
from tvtk.util.ctf import PiecewiseFunction, ColorTransferFunction
from crfsv import segv
import torch
import torch.nn.functional as F

def image_read3d(image_paths): #对OCT sacan 压缩纵轴
    for i, filename in enumerate(image_paths):
        if i == 0:
            image = imageio.imread(filename)
            img_np = np.array(image)
            img3d = np.zeros([len(image_paths), img_np.shape[0], img_np.shape[1]])
            img3d[i] = img_np
        else:
            image = imageio.imread(filename)
            img_np = np.array(image)
            img3d[i] = img_np

    return img3d[:,120:,:]   #[:,120:520,:] [:,170:570,:] [:,150:550,:] 越往下越大

datapath = r"E:\task8\GAMMA\training_data\multi-modality_images\0004\0004"
# datapath = r"E:\task8\OCT500\OCTA-600\OCT\10001"
scanlist = os.listdir(datapath)  # 读取子文件夹下的所有ct切片数
scanlist = natsort.natsorted(scanlist)
for i in range(0, len(scanlist)):  # 读取子文件夹下的所有ct切片路径
    # print(scanlist[i])
    scanlist[i] = os.path.join(datapath, scanlist[i])

image = image_read3d(scanlist)

# 使用interpolate进行resize
input = torch.from_numpy(image[np.newaxis,np.newaxis])
resized_input = F.interpolate(input, size=(400, 400, 400), mode='trilinear', align_corners=False)
image = resized_input.numpy()[0,0]

image = np.transpose(image, (1, 2, 0))
image2 = np.transpose(image, (2, 1, 0))
image2 = np.flip(image2, axis=2)
image2 = np.flip(image2, axis=1)

# 体绘制
mlab.figure(bgcolor=(1, 1, 1))
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(image2), name='3d-ultrasound')   #image*(spix==1)
ctf = ColorTransferFunction()  # 该函数决定体绘制的颜色、灰度等
vol._ctf = ctf
vol._volume_property.set_color(ctf)  # 进行更改，体绘制的colormap及color
vol.update_ctf = True
mlab.show()

seg = segv(
        image,
        sigma=1, #用于预处理的高斯平滑核的宽度。
        n_segments=125,
        max_iter=10,
        compactness=80, #平衡颜色接近性和空间接近性。较高的值给 更加重视空间接近性，使超级像素的形状更加 方形/立方体。
        start_label=1,
        multichannel=False,
        min_size_factor =0.5 #要移除的最小线段尺寸与假定线段尺寸的比例 ``深度*宽度*高度/n_segments'``的比例。
)

# # seg= f['arr_1']
# for ix in range(np.max(seg)):
#         i = ix + 1
#         mlab.figure(bgcolor=(1, 1, 1))
#         seg2 = np.where(seg == i, 1, 0)        #如果等于i，则置为255，否则为0
#         print(i)
#         print(np.mean(np.argwhere(seg == i),axis=0))    #求坐标平均值
#         seg3 = seg2 * image
#         seg3 = np.transpose(seg3, (2, 1, 0))
#         seg3 = np.flip(seg3, axis=2)
#         seg3 = np.flip(seg3, axis=1)
#         seg3 = seg3.astype(np.float32)
#         # 体绘制
#         mlab.figure(bgcolor=(1, 1, 1))
#         vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(seg3), name='3d-ultrasound')  # image*(spix==1)
#         ctf = ColorTransferFunction()  # 该函数决定体绘制的颜色、灰度等
#         vol._ctf = ctf
#         vol._volume_property.set_color(ctf)  # 进行更改，体绘制的colormap及color
#         vol.update_ctf = True
#         # mlab.show()
#
#         fn = "spout2/"+str(i)+'.png'
#         mlab.savefig(filename=fn)
#         # mlab.show()

#计算每个超像素的平均灰度值
avg_gray = []
for i in range(1, np.max(seg) + 1):
    avg_gray.append(np.mean(image[seg == i]))
# 筛选平均灰度值最大的前25个超像素
top_indices = sorted(range(len(avg_gray)), key=lambda x: avg_gray[x], reverse=True)[25:]
# 统计超像素x坐标均值

seg2 = seg
selectindices = list(range(26, 51))
for ti in range(1,126):
    if ti not in selectindices:
        seg2 = np.where(seg2 == ti, 0, seg2)

seg2 = np.transpose(seg2, (2, 1, 0))
seg2 = np.flip(seg2, axis=2)
seg2 = np.flip(seg2, axis=1)
seg2 = seg2.astype(np.float32)

# 假设超像素分割结果储存在三维数组中(seg)
# seg为整型数组，值为1-超像素数
# 将超像素的像素值缩放到[0, 255]
# seg = (seg / np.max(seg) * 255).astype(np.uint8)

# 创建 Mayavi 场景对象
# mlab.options.offscreen = True  # 在没有 GUI 的情况下运行 Mayavi
mlab.figure(bgcolor=(1, 1, 1)) #背景设为白色，否则默认为灰色

# 在场景中绘制超像素分割结果
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(seg2),
        vmin=0, vmax=np.max(seg2), name="Volume")

# 配置颜色和透明度插值函数 (otf) 设置固定的透明度 越低越透明
otfv = 0.05
otf = PiecewiseFunction()
otf.add_point(0, 0.001)
otf.add_point(1, otfv)
otf.add_point(np.max(seg2) * 0.3, otfv)
otf.add_point(np.max(seg2) * 0.6, otfv)
otf.add_point(np.max(seg2), otfv)
vol._otf = otf
vol._volume_property.set_scalar_opacity(otf)

# 配置颜色映射函数 (ctf) 默认的颜色分布设置
ctf = ColorTransferFunction()
ctf.add_rgb_point(0.0, 0.0, 0.0, 0.0)
ctf.add_rgb_point(np.max(seg2), 1.0, 1.0, 1.0)
vol._ctf = ctf

mlab.show()
