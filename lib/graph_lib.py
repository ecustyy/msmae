import dgl
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
from copy import deepcopy
import itertools
import warnings
warnings.filterwarnings('ignore')
import networkx as nx
from sklearn.metrics import confusion_matrix
from lib.test_lib import test_class
from tensorboardX import SummaryWriter

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

def build_graph(xall, attentions):
    bs = xall.shape[0]
    dim = xall.shape[2]
    gs = []
    # 只对整体自注意矩阵处理   xall[28,257,512]
    # 筛选贡献值最大的16个token
    vala, inda = attentions[:, 0, 0, 1:].topk(k=16, dim=1, largest=True, sorted=False)  # 最大的16个值的索引
    # 先排序
    indas, _ = torch.sort(inda)
    # 分别处理每个batch
    for bi in range(bs):
        # 按原索引的先后添加贡献值最大的16个token的特征
        res = torch.zeros(17, dim)
        res[1:, :] = torch.index_select(xall[bi], dim=0, index=indas[bi])
        # 添加cls_token的特征
        res[0, :] = xall[bi, 0, :]
        # 计算token间的互注意值
        dots = torch.matmul(res, res.transpose(-1, -2))
        # 确定每条边都大于零
        dotsz = torch.where(dots > 0.5, dots, torch.Tensor([0]))
        # 对角线归零
        ax = (torch.ones(dots.shape[-1]) - torch.eye(dots.shape[-1]))
        dotsze = dotsz * ax  # 除去对角线的整体自注意矩阵
        # 再次稀疏，筛选一半的互注意值
        dot1 = dotsze[1:, 1:]
        dot2 = dot1.reshape(-1)
        vala2, inda2 = dot2.topk(k=20, dim=0, largest=True, sorted=False)
        a = torch.zeros_like(dot1)  # a是邻接矩阵除去cls_token
        for inx in inda2:
            a[inx // 16, inx % 16] = 1
        # 考虑cls_token
        a2 = torch.zeros_like(dotsze)  # a2是可以输出的邻接矩阵
        a2[1:, 1:] = a
        a2[1:, 0] = torch.Tensor([1])  # 为cls_token添加与所有token的边
        # 为非零值添加边
        # at = a2 + torch.eye(a2.shape[-1])
        # at = torch.triu(a2, diagonal=1)  #提取a的上三角矩阵
        xe, ye = torch.where(a2 > 0)
        # 构建边特征矩阵
        efea = torch.zeros(len(xe), dim)
        for ei in range(len(xe)):
            efea[ei] = torch.abs(res[xe[ei]] - res[ye[ei]])
        # 构建graph
        g = dgl.DGLGraph((xe, ye))
        g.ndata['feat'] = res  # 添加节点特征
        g.edata['feat'] = torch.zeros_like(efea)  # 添加边特征 [36, 512]
        gs.append(g)
        # 画图
        # nx_G = g.to_networkx()
        # pos = nx.kamada_kawai_layout(nx_G)
        # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

    # 整合成一个batch
    batch_graphs = dgl.batch(gs)
    device = torch.device("cuda:0")
    batch_graphs = batch_graphs.to(device)
    batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
    batch_e = batch_graphs.edata['feat'].to(device)

    return batch_graphs, batch_x, batch_e

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
    avg_kappa_list = []

    for epoch in range(n_epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            labels = data[0].to(device=device, dtype=torch.long)
            batch_graphs, batch_x, batch_e = build_graph(data[1], data[2])
            # Graph-ViT 和 输出头
            cls_inda = torch.arange(0, batch_x.shape[0], int(batch_x.shape[0] / len(labels)))
            logits = model.forward(batch_graphs, batch_x, batch_e, cls_inda)
            loss = criterion(logits, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            for p, l in zip(logits.cpu().detach().numpy().argmax(1), labels.cpu().numpy()): #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
                avg_kappa_list.append([p, l])   #前一个是预测标签，后一个是真实值

            loss.backward()
            if torch.isnan(loss):
                print('出错')
            # for name, param in model.named_parameters():
            #     shape, c = (param.grad.shape, param.grad.sum()) if param.grad is not None else (None, None)
            #     print(f'{name}: {param.shape} \n\t grad: {shape} \n\t {c}')
            optimizer.step()
            optimizer.zero_grad()


        _, val_metrics = test(model, val_loader, criterion, device)

        # tensorboard logger
        logger.add_scalar('loss', val_metrics['mean_loss'], epoch)

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

        if np.isnan(val_metrics['mean_loss']):
            print('出错')

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

def test_me(model, val_loader, savename='output/savedata'):#测试函数

    pres = []
    device = 'cuda:0'
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            labels = data[0].to(device=device, dtype=torch.long)    #真值
            batch_graphs, batch_x, batch_e = build_graph(data[1], data[2])  #节点特征值，所有注意值
            # Graph-ViT 和 输出头
            cls_inda = torch.arange(0, batch_x.shape[0], int(batch_x.shape[0] / len(labels)))
            outputs = model.forward(batch_graphs, batch_x, batch_e, cls_inda)
            pres.append(outputs)

    # 保存所有数据
    np.savez(savename + '/graph-data', val_loader.dataset[0], torch.cat(pres).cpu())
    class_name = ['Normal', 'CNV', 'DR', 'AMD', 'CSC', 'RVO', 'Others']
    test_class(class_name, val_loader.dataset[0], torch.cat(pres).cpu(), path_experiment=savename + '/')


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

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            # load batch
            labels = data[0].to(device=device, dtype=torch.long)
            # one hot encoding for AUC
            bs = labels.size()[0]
            num_classes = 7
            onehot = (labels.reshape(bs, 1) == torch.arange(num_classes).reshape(1, num_classes).to(device)).float()

            batch_graphs, batch_x, batch_e = build_graph(data[1], data[2])
            # Graph-ViT 和 输出头
            cls_inda = torch.arange(0, batch_x.shape[0], int(batch_x.shape[0] / len(labels)))
            outputs = model.forward(batch_graphs, batch_x, batch_e, cls_inda)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
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