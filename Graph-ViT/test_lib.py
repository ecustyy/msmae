
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

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

from sklearn.metrics import confusion_matrix
import itertools

def test_me(model, val_loader, class_name, path_experiment='output/'):#测试函数

    gt = []
    predict = []
    criterion = nn.CrossEntropyLoss()
    device = 'cuda:0'
    pres = []
    xalls = []
    attalls = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # load batch
            fundus_imgs = data[0].to(device=device, dtype=torch.float32)
            #oct_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].to(device=device, dtype=torch.long)
            # one hot encoding for AUC
            pre, xall, attentions = model(fundus_imgs, True)    #[7,17,257,257]
            pres.append(pre)
            xalls.append(xall)
            attalls.append(attentions)
            if i == 0:
                fnlist = data[3]
            else:
                for ic in range(len(data[3])):
                    fnlist.append( data[3][ic])


            nh = attentions.shape[1]  # number of head

            for i in range(attentions.shape[0]):    #对batch中的图像索引
                output_dir = os.path.join('output2', data[3][i])
                os.makedirs(output_dir, exist_ok=True)
                # 输出未经处理的所有自注意结果：只整体的
                nfi = nn.functional.interpolate(attentions[i,0].unsqueeze(0).unsqueeze(0), scale_factor=5,
                                                   mode="nearest")[0][0].cpu().numpy()
                fname = os.path.join(output_dir, "ori-all" + ".png")
                plt.imsave(fname=fname, arr=nfi, format='png')
                print(f"{fname} saved.")
                # 对多头自注意Softmax处理
                sh_att = nn.Softmax(dim=-1)(attentions[i,1:,1:,1:])   #去掉cls_token, 对所有的行进行归一化

                # 对多头自注意还原成2维
                sh_con = sh_att[:, 0, :].reshape(nh-1, -1)  # 只保留与clstoken的注意力，为1维的

                # 所有多头自注意提取的mask，多头关注的patch在原图对应位置显示
                max_con, _ = torch.max(sh_con, dim=-2)  #为每个patch在所有多头中提取最大贡献
                max_con16 = max_con.reshape(16, 16)  # 恢复成2维数据
                nfi = nn.functional.interpolate(max_con16.unsqueeze(0).unsqueeze(0), scale_factor=25, mode="nearest")[0][0].cpu().numpy()  # 插值和上采样，按patchsize倍数扩大，[6,224,224]
                fname = os.path.join(output_dir, "all-deal1" + ".png")
                plt.imsave(fname=fname, arr=nfi, format='png')
                print(f"{fname} saved.")

                # 对整体自注意矩阵处理
                altt = attentions[i, 0].cpu()   #整体的自注意矩阵
                almin = torch.min(altt)
                alttz = altt - almin  #最小值为0的整体的自注意矩阵
                ax = (torch.ones(altt.shape[-1]) - torch.eye(altt.shape[-1]))
                aleye = alttz * ax  #除去对角线的整体自注意矩阵

                # 所有多头自注意提取的mask
                vala, inda = max_con.topk(k=16, dim=0, largest=True, sorted=False)  #最大的16个值的索引
                xin, yin = torch.meshgrid(inda, inda)
                altt1 = aleye[1:,1:]    #除去cls_token
                a = torch.zeros_like(altt1)
                value = torch.Tensor([1])
                a.index_put_([xin, yin], value)
                alcross = altt1 * a   #Cross Attention Matrix
                alcross = alcross / alcross.max()  # 归一化
                nfi = nn.functional.interpolate(alcross.unsqueeze(0).unsqueeze(0), scale_factor=5,
                                                 mode="nearest")[0][0].cpu().numpy()
                fname = os.path.join(output_dir, "all-deal3" + ".png")
                plt.imsave(fname=fname, arr=nfi, format='png')
                print(f"{fname} saved.")

                # 提取最大16个贡献的token合并到小的整体自注意矩阵
                indas,_ = torch.sort(inda)
                alcross16 = alcross[torch.meshgrid(indas,indas)]  #根据head注意的patch筛选的互注意矩阵，合并到小图
                nfi = nn.functional.interpolate(alcross16 .unsqueeze(0).unsqueeze(0), scale_factor=25,
                                                mode="nearest")[0][0].cpu().numpy()
                fname = os.path.join(output_dir, "all-deal3(2)" + ".png")
                plt.imsave(fname=fname, arr=nfi, format='png')
                print(f"{fname} saved.")

                # 提取最大16个贡献的token映射到原图尺寸
                a = torch.zeros_like(max_con)
                a[indas] = 1
                max16_con = max_con * a
                max16_con16 = max16_con.reshape(16, 16)  # 恢复成2维数据
                nfi = nn.functional.interpolate(max16_con16.unsqueeze(0).unsqueeze(0), scale_factor=25, mode="nearest")[0][
                    0].cpu().numpy()  # 插值和上采样，按patchsize倍数扩大，[6,224,224]
                fname = os.path.join(output_dir, "all-deal3(3)" + ".png")
                plt.imsave(fname=fname, arr=nfi, format='png')
                print(f"{fname} saved.")

                # 保存对原图的蒙版
                f1 = fundus_imgs[i].cpu()
                f2 = f1.transpose(0, 2).numpy()
                f = np.zeros([400, 400, 4])
                f[:, :, :3] = f2
                f[:, :, 3] = nfi
                fname = os.path.join(output_dir, "all-deal3(4)" + ".png")
                plt.imsave(fname=fname, arr=f, format='png')
                print(f"{fname} saved.")

                # 再次稀疏，筛选一半的互注意值
                alcross1d = alcross.reshape(-1)
                vala2, inda2 = alcross1d.topk(k=120, dim=0, largest=True, sorted=False)
                a = torch.zeros_like(alcross)
                for inx in inda2:
                    a[inx//256, inx%256]=1
                alcrossm = alcross * a

                # 画边：将所有的有效边添加在图中
                plt.clf()
                an = alcrossm.numpy()
                edges = np.where(an > 0)
                for ei in range(len(edges[0])):
                    xs = (edges[0][ei]//16) * 25 + 13
                    ys = (edges[0][ei]%16) * 25 + 13
                    xe = (edges[1][ei]//16) * 25 + 13
                    ye = (edges[1][ei]%16) * 25 + 13
                    plt.plot([ys, ye], [xs, xe], color="red", linewidth=1)
                plt.axis('off')
                plt.imshow(f)
                fname = os.path.join(output_dir, "all-deal4(1)" + ".png")
                plt.savefig(fname=fname, format='png', bbox_inches='tight')

                # 对筛选的alcross16再次稀疏，筛选一半的互注意值
                alcross1d = alcross16.reshape(-1)
                vala3, inda3 = alcross1d.topk(k=120, dim=0, largest=True, sorted=False)
                a = torch.zeros_like(alcross16)
                for inx in inda3:
                    a[inx // 16, inx % 16] = 1
                alcross16m = alcross16 * a
                nfi = nn.functional.interpolate(alcross16m.unsqueeze(0).unsqueeze(0), scale_factor=25,
                                                mode="nearest")[0][0].cpu().numpy()
                fname = os.path.join(output_dir, "all-deal4(2)" + ".png")
                plt.imsave(fname=fname, arr=nfi, format='png')
                print(f"{fname} saved.")

            outputs = model(fundus_imgs)
            predicted = torch.max(outputs.data, dim=1).indices
            for idx, _ in enumerate(labels):
                gt.append(labels[idx])
                predict.append(predicted[idx])

    # 保存所有数据
    np.savez("output/all-data.npz", fnlist, torch.stack(pres).cpu(), torch.stack(xalls).cpu(), torch.stack(attalls).cpu())

    ground_truth = np.array([g.item() for g in gt])
    prediction = np.array([pred.item() for pred in predict])
    title='Confusion matrix'
    cm = confusion_matrix(ground_truth, prediction)
    cmap=plt.cm.Blues
    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_name))
    plt.xticks(tick_marks, class_name, rotation=45)
    plt.yticks(tick_marks, class_name)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path_experiment + "confusion matrix.eps", format='eps', dpi=100, bbox_inches='tight')
    plt.savefig(path_experiment + "confusion matrix.jpg", bbox_inches='tight')
    plt.show()

    file_perf = open(path_experiment + 'performances.txt', 'w')
    # Sensitivity, Specificity and F1 per class
    print('class acc, sen, spe, pre, miou, f1')
    file_perf.write('class acc, sen, spe, pre, miou, f1' + '\n')
    n_classes = len(class_name)
    allacc = []
    allsen = []
    allspe = []
    allpre = []
    allmiou = []
    allf1 = []
    for i in range(n_classes):
        y_test = [int(x == i) for x in ground_truth]  # obtain binary label per class
        tn, fp, fn, tp = confusion_matrix(y_test, [int(x == i) for x in prediction]).ravel()
        acc = float(tp + tn) / (tn + fp + fn + tp)
        sen = float(tp) / (fn + tp)
        spe = float(tn) / (tn + fp)
        pre = float(tp) / (tp + fp)
        miou = float(tp) / (tp + fp + fn)
        f1 = 2 * pre * sen / (pre + sen)
        print(class_name[i], '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (acc, sen, spe, pre, miou, f1))
        file_perf.write(class_name[i]+ '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (acc, sen, spe, pre, miou, f1) + '\n')
        allacc.append(acc)
        allsen.append(sen)
        allspe.append(spe)
        allpre.append(pre)
        allmiou.append(miou)
        allf1.append(f1)
    aacc = sum(allacc) / n_classes
    asen = sum(allsen) / n_classes
    aspe = sum(allspe) / n_classes
    apre = sum(allpre) / n_classes
    amiou = sum(allmiou) / n_classes
    af1 = sum(allf1) / n_classes
    print('mean_of_all', '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (aacc, asen, aspe, apre, amiou, af1))
    file_perf.write('mean_of_all' + '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (aacc, asen, aspe, apre, amiou, af1) + '\n')

    file_perf.close()
    print("done")
