
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
from efficientnet_pytorch import EfficientNet

#model = EfficientNet.from_name(args.arch)
#model = EfficientNet.from_pretrained('efficientnet-b0')
#from torchvision import transforms, datasets
import torchvision.transforms as trans
from torch.utils.data import Dataset
from copy import deepcopy
import itertools
import warnings
warnings.filterwarnings('ignore')


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
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
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

        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        if fundus_img.shape[0] == 2000:
            fundus_img = fundus_img[1000-967:1000+967, 1496-978:1496+978, :]    #裁剪过大图片

        fundus_img = (fundus_img / 255.).astype("float32")

        if self.img_transforms is not None:
            fundus_img = fundus_img.copy()
            fundus_img = self.img_transforms(fundus_img)

        # normlize on GPU to save CPU Memory and IO consuming.
        # fundus_img = (fundus_img / 255.).astype("float32")
        # oct_img = (oct_img / 255.).astype("float32")

        #fundus_img = fundus_img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        #oct_img = oct_img.squeeze(-1)  # D, H, W, 1 -> D, H, W

        if self.mode == 'test':
            return fundus_img, real_index
        if self.mode == "train":
            label = label.argmax()
            label = np.min((label, 1))
            label = label.astype('int64')
            return fundus_img, label, real_index

    def __len__(self):
        return len(self.file_list)

# Network
class Model(nn.Module):#双流网络模型
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """

    def __init__(self):
        super(Model, self).__init__()
        #self.fundus_branch =  EfficientNetB3()
        self.fundus_branch =  EfficientNet.from_pretrained('efficientnet-b3')
        self.oct_branch = resnet34(pretrained=False, num_classes=512)
        self.decision_branch = nn.Linear(1512 , 3)  # ResNet34 use basic block, expansion = 1

        # replace first conv layer in oct_branch
        self.oct_branch.conv1 = torch.nn.Conv2d(256, 64,
                                          kernel_size=7,
                                          stride=2,
                                          padding=3)

    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img) #[b,1000]
        b2 = self.oct_branch(oct_img)   #[b,512]
        b1 = torch.flatten(b1, 1)
        b2 = torch.flatten(b2, 1)
        logit = self.decision_branch(torch.cat([b1, b2], 1))

        return logit

def train_ID(model, train_loader, val_loader, criterion, optimizer, n_epochs, device):#训练函数
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
    best_model = deepcopy(model)
    val_best_loss = np.inf
    history_train = {'acc': [], 'loss': []}
    history_val = {'acc': [], 'loss': []}

    model.train()
    avg_loss_list = []
    avg_kappa_list = []
    best_kappa = 0.
    for epoch in range(n_epochs):
        model.train() #train 与 eval 是 module 中对立的两种模式；dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中；训练时model.train()，测试时model.eval()
        for i, data in enumerate(train_loader, 0):
            # load batch
            fundus_imgs = data[0].to(device=device, dtype=torch.float32)
            oct_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].to(device=device, dtype=torch.long)

            logits = model(fundus_imgs, oct_imgs)
            loss = criterion(logits, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            for p, l in zip(logits.cpu().detach().numpy().argmax(1), labels.cpu().numpy()): #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
                avg_kappa_list.append([p, l])   #前一个是预测标签，后一个是真实值

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        _, val_metrics = test(model, val_loader, criterion, device)
        _, train_metrics = test(model, train_loader, criterion, device)

        print('Train |Epoch %i: loss = %f | accuracy: %f | balanced accuracy = %f'
              % (epoch, train_metrics['mean_loss'],
                 train_metrics['accuracy'],
                 train_metrics['accuracy']))
        print('Validation |Epoch %i: loss = %f | accuracy: %f | balanced accuracy = %f'
              % (epoch, val_metrics['mean_loss'],
                 val_metrics['accuracy'],
                 val_metrics['accuracy']))
        print('\n')

        # history
        history_val['acc'].append(val_metrics['accuracy'])
        history_val['loss'].append(val_metrics['mean_loss'])
        history_train['acc'].append(train_metrics['accuracy'])
        history_train['loss'].append(train_metrics['mean_loss'])

        if val_metrics['mean_loss'] < val_best_loss:
            best_model = deepcopy(model)
            val_best_loss = val_metrics['mean_loss']
            torch.save(best_model.state_dict(), 'model_dr.pth')
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

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            # load batch
            fundus_imgs = data[0].to(device=device, dtype=torch.float32)
            oct_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].to(device=device, dtype=torch.long)
            # one hot encoding for AUC
            bs = labels.size()[0]
            num_classes = 3
            onehot = (labels.reshape(bs, 1) == torch.arange(num_classes).reshape(1, num_classes).to(device)).float()

            outputs = model(fundus_imgs, oct_imgs)
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
    results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)

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
