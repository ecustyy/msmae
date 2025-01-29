"""
Create A Data Dictionary
"""

import os
import natsort
import xlrd
import numpy as np
import imageio


dict=['NORMAL', 'CNV', 'DR','AMD','CSC','RVO','OTHERS'] #前四个是3M中使用的，

# 从文件夹遍历所有子文件夹的子文件
#将文件路径保存在datasetlist
#datasetlist['train'][modal][ct] modal是数据类别 ct指B扫描数
def read_datasetme(data_dir,modality,saveroot): #数据路径，模式{集合} 测试比例 分别读取训练集和评估集文件路径
    if not os.path.exists(os.path.join(saveroot,"data6n.npz")):
        #如果没有找到保存的hdf5文件
        #读取文件路径和标签信息
        id_cols, disease_cols = read_excel()
        dedi = np.zeros(len(disease_cols))
        for id in range(len(disease_cols)):
            dedi[id] = dict.index(disease_cols[id])
        classnum = int(dedi.max()+1)
        clist = []
        for i in range(classnum):
            clist.append([])

        for cnt, element in enumerate(dedi):
            clist[int(element)].append(cnt)

        datasetlist = {}
        for modal in modality:  # OCT/OCTA/Label
            datasetlist.update({modal: {}})  # update增加子列表

        for cnt,ct in enumerate(id_cols[:]): #cnt表示索引，ct表示元素
            for modal in modality:
                if modal == 'OCT' or modality == 'OCTA': #读取三维数据
                    datasetlist[modal].update({cnt: {}})
                    scanlist = os.listdir(os.path.join(data_dir, modal, ct))  # 读取子文件夹下的所有ct切片数
                    scanlist = natsort.natsorted(scanlist)
                    for i in range(0, len(scanlist)):  # 读取子文件夹下的所有ct切片路径
                        scanlist[i] = os.path.join(data_dir, modal, ct, scanlist[i])
                    datasetlist[modal][cnt] = scanlist  # 最后一个维度是保存list
                else:
                    datasetlist[modal].update({cnt: {}})
                    labeladdress = os.path.join(data_dir, modal, ct)
                    datasetlist[modal][cnt] = labeladdress + '.bmp'  # 最后一个维度是保存单个文件路径

        #根据多模态路径读取图像
        data2d = read_imageme(datasetlist)

        Bclist = []
        for i in range(classnum):
            Bzl = []
            for ct in clist[i]: #cnt表示索引，ct表示元素
                for bct in datasetlist['OCT'][ct]:
                    Bzl.append(bct)
            Bclist.append(Bzl)

        #保存图像和标签
        np.savez(os.path.join(saveroot, "data6n.npz"), data2d, datasetlist, dedi, clist, Bclist, id_cols)
    else:
        print("found pickle !!!")
        f = np.load(os.path.join(saveroot, "data6n.npz"),allow_pickle=True)
        data2d = f['arr_0']
        datasetlist = f['arr_1']
        dedi = f['arr_2']
        clist = f['arr_3']
        Bclist = f['arr_4']
        id_cols = f['arr_5']

    return data2d, datasetlist, dedi, clist, Bclist, id_cols


def read_excel():
    # 打开文件
    workBook = xlrd.open_workbook(r'D:\task8\logs\Text labels6.xlsx')

    sheet1_content1 = workBook.sheet_by_index(0)  # sheet索引从0开始
    # 3. sheet的名称，行数，列数
    print(sheet1_content1.name, sheet1_content1.nrows, sheet1_content1.ncols)

    # 4. 获取整行和整列的值（数组）
    id_cols = sheet1_content1.col_values(0)  # 获取第1列内容
    id_cols.pop(0) #去除标题
    id_cols= [int(i) for i in id_cols]
    id_cols = [str(i) for i in id_cols]
    disease_cols = sheet1_content1.col_values(4)  # 获取第5列内容
    disease_cols.pop(0)
    return id_cols, disease_cols
    #读图像并保存hdf5文件 self.data[modality_num,r,c,scan_num,ct_num] self.label[0,r,c,ct_num]

def read_imageme(datasetlist):
    data2d = np.zeros((1, 400, 400, len(datasetlist['FULL'])), dtype=np.int)
    print("picking ...It will take some minutes")

    ctlist = list(datasetlist['FULL'])  # 读取二维图像
    ct_num = -1
    for ct in ctlist:
        ct_num += 1
        labeladress = datasetlist['FULL'][ct]
        data2d[0, :, :, ct_num] = image_transform(labeladress)

    return data2d

# 读取图像并转换大小
def image_transform(filename): #对OCT sacan 压缩纵轴
    image = imageio.imread(filename)
    resize_image = np.array(image)
    return resize_image

def read_dataset_post(data_dir,feature_dir,trainids,valids,modality):
    datasetlist = {'train': {}}
    ctlist = os.listdir(os.path.join(data_dir, modality[0]))
    ctlist = natsort.natsorted(ctlist)
    datasetlist['train'].update({'feature': {}})
    datasetlist['train'].update({'label': {}})
    for ct in ctlist[trainids[0]:trainids[1]]:
        datasetlist['train']['feature'].update({ct: {}})
        datasetlist['train']['feature'][ct] = os.path.join(feature_dir,ct+'.npy')
        datasetlist['train']['label'].update({ct: {}})
        datasetlist['train']['label'][ct] = os.path.join(data_dir,modality[-1],ct+'.bmp')
    train_records = datasetlist['train']
    return train_records