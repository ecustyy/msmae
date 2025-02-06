
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34

#model = EfficientNet.from_name(args.arch)
#model = EfficientNet.from_pretrained('efficientnet-b0')
#from torchvision import transforms, datasets
import torchvision.transforms as trans
from torch.utils.data import Dataset
from copy import deepcopy
import itertools
import warnings
warnings.filterwarnings('ignore')
from lib.test_lib import test_class
from tensorboardX import SummaryWriter

class Losstry(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self):
        super(Losstry, self).__init__()
        self.CE = nn.CrossEntropyLoss() #默认为求均值
        self.sf = nn.Softmax(dim=-1)

    def forward(self, logits, attentions, labels):
        hatt = attentions[:,1:,0,1:] #原size[28,17,257,257]，这里只用每个头的自注意结果，第0行表示与cls_token的自注意，最后的1表示除去cls_token的自注意
        shatt = self.sf(hatt)   #对最后一个维度归一化，即贡献值为1
        mv, mi = torch.max(shatt, dim=-1)
        ls = torch.mean(1 - mv) #选取的值尽可能大
        # torch.sum(shatt)-torch.sum(mv)
        mv2, mi2 = torch.max(shatt, dim=-2)
        ld = torch.mean(1 - mv2) # 不同的head关注不同的值,一个patch在不同head中的最大值要大
        lce = self.CE(logits, labels)   #求平均，对于batch和增强的倍数2

        loss = ls+ld+lce
        return loss,lce,ls,ld

# DataLoader
# GAMMA_sub1_dataset:
#
#     Load oct images and fundus images according to `patient id`
#
class GAMMA_sub1_dataset(Dataset):#GAMMA数据集函数
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 img_transforms,
                 oct_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes

        if self.mode == 'train':
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root) ]
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                 key=lambda x: int(x.strip("_")[0]))

        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        if fundus_img.shape[0] == 2000:
            fundus_img = fundus_img[1000-967:1000+967, 1496-978:1496+978, :]    #裁剪过大图片

        '''
        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)
        oct_img = np.zeros((oct_series_0.shape[0], oct_series_0.shape[1], len(oct_series_list)), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[:,:,k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)
        '''
        fundus_img = (fundus_img / 255.).astype("float32")
        #oct_img = (oct_img / 255.).astype("float32")

        if self.img_transforms is not None:
            fundus_img = fundus_img.copy()
            fundus_img = self.img_transforms(fundus_img)
        # if self.oct_transforms is not None:
        #     oct_img = oct_img.copy()
        #     oct_img = self.oct_transforms(oct_img)

        # normlize on GPU to save CPU Memory and IO consuming.
        # fundus_img = (fundus_img / 255.).astype("float32")
        # oct_img = (oct_img / 255.).astype("float32")

        #fundus_img = fundus_img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        #oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W

        if self.mode == 'test':
            return fundus_img, fundus_img, real_index
        if self.mode == "train":
            label = label.argmax()
            label = label.astype('int64')
            return fundus_img, fundus_img, label

    def __len__(self):
        return len(self.file_list)

# Network
class DModel(nn.Module):#双流网络模型
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """

    def __init__(self, mmodel, gmodel):
        super(DModel, self).__init__()
        self.mmodel = mmodel
        self.gmodel = gmodel

    def forward(self, img):
        v, att = self.mmodel(img, True)
        logit = self.gmodel(v)

        return logit

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss

def train_ID(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, save_path):#训练函数
    """
    Method used to train our classifier

    Args:
        model: (nn.Module) the neural network
        train_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images
        optimizer: (torch.optim) an optimization algorithm
        n_epochs: (int) number of epochs performed during training

    Returns:
        best_model: (nn.Module) the trained neural network
    """
    # tensorboard
    logger = SummaryWriter(save_path)  # tensorboard初始化一个写入单元
    best_model = deepcopy(model)
    val_best_loss = np.inf
    history_train = {'acc': [], 'loss': []}
    history_val = {'acc': [], 'loss': []}

    model.train()
    avg_loss_list = []
    avg_kappa_list = []
    best_kappa = 0.
    for epoch in range(n_epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            # load batch
            fundus_imgs = data[0].to(device=device, dtype=torch.float32)
            #oct_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].to(device=device, dtype=torch.long)

            logits,_,attentions = model(fundus_imgs, True)
            loss,lce,ls,ld = criterion(logits, attentions, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            for p, l in zip(logits.cpu().detach().numpy().argmax(1), labels.cpu().numpy()): #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
                avg_kappa_list.append([p, l])   #前一个是预测标签，后一个是真实值

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        _, val_metrics = test(model, val_loader, criterion, device)

        # tensorboard logger
        logger.add_scalar('loss', val_metrics['mean_loss'], epoch)
        logger.add_scalar('lce', val_metrics['mean_lce'], epoch)
        logger.add_scalar('ls', val_metrics['mean_ls'], epoch)
        logger.add_scalar('ld', val_metrics['mean_ld'], epoch)
        # _, train_metrics = test(model, train_loader, criterion, device)
        #
        # print('Train |Epoch %i: loss = %f | accuracy: %f | balanced accuracy = %f'
        #       % (epoch, train_metrics['mean_loss'],
        #          train_metrics['accuracy'],
        #          train_metrics['accuracy']))
        print('Validation |Epoch %i: loss = %f | accuracy: %f '
              % (epoch, val_metrics['mean_loss'],
                 val_metrics['accuracy']))
        print('\n')

        # history
        history_val['acc'].append(val_metrics['accuracy'])
        history_val['loss'].append(val_metrics['mean_loss'])
        # history_train['acc'].append(train_metrics['accuracy'])
        # history_train['loss'].append(train_metrics['mean_loss'])

        if val_metrics['mean_loss'] < val_best_loss:
            best_model = deepcopy(model)
            val_best_loss = val_metrics['mean_loss']
            torch.save(best_model.state_dict(), save_path+"/best_model_{:.4f}".format(val_best_loss) + '.pth')
            print('save bestmodel')

    return best_model, history_train, history_val

def test(model, data_loader, criterion, device):
    """
    Method used to test a CNN

    Args:
        model: (nn.Module) the model
        data_loader: (DataLoader) a DataLoader
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images

    Returns:
        results_df: (DataFrame) the label predicted for every subject
        results_metrics: (dict) a set of metrics
    """
    model.eval()
    columns = ["img_idx", "scores", "proba",
               "true_label", "onehot", "predicted_label"]
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0
    total_lce = 0
    total_ls = 0
    total_ld = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            # load batch
            fundus_imgs = data[0].to(device=device, dtype=torch.float32)
            #oct_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].to(device=device, dtype=torch.long)
            # one hot encoding for AUC
            bs = labels.size()[0]
            num_classes = 3
            onehot = (labels.reshape(bs, 1) == torch.arange(num_classes).reshape(1, num_classes).to(device)).float()

            outputs,_, attentions = model(fundus_imgs, True)
            loss,lce,ls,ld = criterion(outputs, attentions, labels)
            total_loss += loss.item()
            total_lce += lce.item()
            total_ls += ls.item()
            total_ld += ld.item()
            proba = torch.nn.Softmax(dim=1)(outputs)
            predictions, predicted_labels = torch.max(outputs.data, dim=1).values, torch.max(outputs.data, dim=1).indices
            for idx, _ in enumerate(labels):
                row = [idx, proba[idx], predictions[idx], labels[idx], onehot[idx], predicted_labels[idx]]
                row_df = pd.DataFrame([row], columns=columns)
                results_df = pd.concat([results_df, row_df])

    results_metrics = compute_metrics(results_df.scores.values,
                                      results_df.true_label.values,
                                      results_df.onehot.values,
                                      results_df.predicted_label.values
                                      )
    results_df.reset_index(inplace=True, drop=True)
    results_metrics['mean_loss'] = total_loss / 10
    results_metrics['mean_lce'] = total_lce / 10
    results_metrics['mean_ls'] = total_ls / 10
    results_metrics['mean_ld'] = total_ld / 10

    return results_df, results_metrics

def compute_metrics(scores, ground_truth, onehot, prediction):#测试函数中计算指标
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score

    """Computes the accuracy, sensitivity, specificity and balanced accuracy"""
    matches = 0
    for pred, gt in zip(prediction, ground_truth):
      if pred.item() == gt.item():
        matches += 1

    ground_truth = np.array([g.item() for g in ground_truth])
    onehot = np.array([x.data.cpu().numpy() for x in onehot])
    scores = np.array([x.data.cpu().numpy() for x in scores])
    prediction = np.array([pred.item() for pred in prediction])


    metrics_dict = dict()
    metrics_dict['accuracy'] = matches / prediction.size
    metrics_dict['confusion_matrix'] = confusion_matrix(ground_truth, prediction)
    try:
        metrics_dict['AUC'] = roc_auc_score(onehot, scores)
    except:
        metrics_dict['AUC'] = 0
    # one-hot encoding of target labels to compute AUC

    return metrics_dict

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("confusion matrix.jpg")
    plt.show()

# Utils
def train(model, train_loader, val_loader, criterion, optimizer, n_epochs, device):
    model.train()
    avg_loss_list = []
    avg_kappa_list = []
    best_kappa = 0.
    for epoch in range(n_epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            # load batch
            fundus_imgs = data[0].to(device=device, dtype=torch.float32)
            #oct_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].to(device=device, dtype=torch.long)

            logits = model(fundus_imgs)
            loss = criterion(logits, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            for p, l in zip(logits.cpu().detach().numpy().argmax(1), labels.cpu().numpy()):
                avg_kappa_list.append([p, l])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad() #清空过往梯度，为下一波梯度累加做准备,否则梯度会在之后的iter进行累加

            avg_loss_list.append(loss.cpu().detach().numpy())

            avg_loss = np.array(avg_loss_list).mean()
            avg_kappa_list = np.array(avg_kappa_list)
            avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1], weights='quadratic')
            avg_loss_list = []
            avg_kappa_list = []
            print("[TRAIN] epoch={} iter={} avg_loss={:.4f} avg_kappa={:.4f}".format(epoch, i, avg_loss, avg_kappa))


        avg_loss, avg_kappa = val(model, val_loader, criterion, device)
        print("[EVAL] epoch={} avg_loss={:.4f} kappa={:.4f}".format(epoch, avg_loss, avg_kappa))
        if avg_kappa >= best_kappa:
            best_kappa = avg_kappa
            torch.save(model.state_dict(), "best_model_{:.4f}".format(best_kappa)+'.pth')
        model.train()

def val(model, val_loader, criterion, device):
    model.eval()
    avg_loss_list = []
    cache = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # load batch
            fundus_imgs = data[0].to(device=device, dtype=torch.float32)
            #oct_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].to(device=device, dtype=torch.long)

            logits = model(fundus_imgs)
            for p, l in zip(logits.cpu().detach().numpy().argmax(1), labels.cpu().numpy()):
                cache.append([p, l])

            loss = criterion(logits, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            avg_loss_list.append(loss.cpu().detach().numpy())
    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
    avg_loss = np.array(avg_loss_list).mean()

    return avg_loss, kappa

def test_cfp_mask(model, val_loader, savename='output/cfp-mask-data'):#测试函数

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
                output_dir = os.path.join(savename, data[3][i])
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

    # 保存所有数据
    np.savez(savename+'/cfp-mask-data', val_loader.dataset[1], torch.cat(pres).cpu(), torch.cat(xalls).cpu(), torch.cat(attalls).cpu(), fnlist)

def test_cfp_mask2(model, val_loader, savename='output/cfp-mask-data'):#测试函数

    device = 'cuda:0'
    pres = []
    xalls = []
    attalls = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # load batch
            fundus_imgs = data[0].to(device=device, dtype=torch.float32)
            pre, xall, attentions = model(fundus_imgs, True)    #[7,17,257,257]
            pres.append(pre)
            xalls.append(xall)
            attalls.append(attentions)
            if i == 0:
                fnlist = data[3]
            else:
                for ic in range(len(data[3])):
                    fnlist.append( data[3][ic])

    # 保存所有数据
    np.savez(savename+'/cfp-mask-data', val_loader.dataset[1], torch.cat(pres).cpu(),fnlist)
    class_name = ['Normal', 'CNV', 'DR', 'AMD', 'CSC', 'RVO', 'Others']
    test_class(class_name, val_loader.dataset[1], torch.cat(pres).cpu(), path_experiment=savename+'/')