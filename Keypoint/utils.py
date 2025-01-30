import os
import copy
import torch
import shutil
import importlib
import numpy as np
import numpy as np
from metrics import accuracy_score, confusion_matrix, sensitivity_score, specificity_score, f1_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(config_filename):
    config_path = "configs.{}".format(config_filename.split('.')[0])
    module = importlib.import_module(config_path)
    return module.Config()


def splitprint():
    print("#"*100)


def runid_checker(opts, if_syn=False):
    rootpath = opts.train_collection
    if if_syn:
        rootpath = opts.syn_collection
    valset_name = os.path.split(opts.val_collection)[-1]
    config_filename = opts.model_configs
    run_id = opts.run_id
    target_path = os.path.join(rootpath, "models", valset_name, config_filename, "run_" + str(run_id))
    if os.path.exists(target_path):
        if opts.overwrite:
            shutil.rmtree(target_path)
        else:
            print("'{}' exists!".format(target_path))
            return False
    os.makedirs(target_path)
    print("checkpoints are saved in '{}'".format(target_path))
    return True


def predict_dataloader(model, loader, device):
    model.eval()
    predicts = np.array([])
    scores = np.array([])
    expects = np.array([])
    for i, data in enumerate(loader, 0):  # 从0开始索引
        # load batch
        fundus_imgs = data[0].to(device=device, dtype=torch.float32)
        # oct_imgs = data[1].to(device=device, dtype=torch.float32)
        label = data[1].to(device=device, dtype=torch.long)

        outputs = model(fundus_imgs)
        output = np.squeeze(torch.softmax(outputs, dim=1).cpu().detach().numpy())
        predicts = np.append(predicts, np.argmax(output, axis=1))
        scores = np.append(scores, np.amax(output, axis=1))
        expects = np.append(expects, label.cpu().numpy())
    return predicts, scores, expects

def test_dataloader(model, loader, device):
    model.eval()
    predicts = np.array([])
    scores = np.zeros((4,3))
    expects = np.array([])
    for i, data in enumerate(loader, 0):  # 从0开始索引
        # load batch
        fundus_imgs = data[0].to(device=device, dtype=torch.float32)
        oct_imgs = data[1].to(device=device, dtype=torch.float32)
        label = data[2].to(device=device, dtype=torch.long)

        outputs = model(fundus_imgs, oct_imgs)
        output = np.squeeze(torch.softmax(outputs, dim=1).cpu().detach().numpy())
        predicts = np.append(predicts, np.argmax(output, axis=1))
        if i == 0:
            scores = output
        else:
            scores = np.vstack((scores, output))
        expects = np.append(expects, label.cpu().numpy())
    return predicts, scores, expects

def batch_eval(predicts, expects, cls_num=3, verbose=False):
    def multi_to_binary(Y, pos_cls_idx):
        Y_cls = copy.deepcopy(np.array(Y))
        pos_idx = np.where(np.array(Y) == pos_cls_idx)
        neg_idx = np.where(np.array(Y) != pos_cls_idx)
        Y_cls[neg_idx] = 0
        Y_cls[pos_idx] = 1
        return Y_cls

    metrics = {"overall": {}, "0": {}, "1": {}, "2": {}}
    cls_list = ["0", "1", "2"]
    metrics["overall"]["accuracy"] = accuracy_score(expects, predicts)
    metrics["overall"]["confusion_matrix"] = confusion_matrix(expects, predicts)
    # metrics per class
    for cls_idx in range(cls_num):
        cls_name = cls_list[cls_idx]
        predicts_cls = multi_to_binary(predicts, cls_idx)
        expects_cls = multi_to_binary(expects, cls_idx)
        try:
            sen = sensitivity_score(expects_cls, predicts_cls)
        except:
            sen = 0

        try:
            spe = specificity_score(expects_cls, predicts_cls)
        except:
            spe = 0

        try:
            f1 = f1_score(sen, spe)
        except:
            f1 = 0
        metrics[cls_name]["sensitivity"] = sen
        metrics[cls_name]["specificity"] = spe
        metrics[cls_name]["f1_score"] = f1
    metrics["overall"]["f1_score"] = np.average(
        [metrics[cls_name]["f1_score"] for cls_name in cls_list])

    if verbose:
        print(" Class\tSen.\tSpe.\tF1score\n",
              "0\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\n".format(
                  sen=metrics["0"]["sensitivity"],
                  spe=metrics["0"]["specificity"],
                  f1=metrics["0"]["f1_score"]),
              "1\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\n".format(
                  sen=metrics["1"]["sensitivity"],
                  spe=metrics["1"]["specificity"],
                  f1=metrics["1"]["f1_score"]),
              "2\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\n".format(
                  sen=metrics["2"]["sensitivity"],
                  spe=metrics["2"]["specificity"],
                  f1=metrics["2"]["f1_score"]),
              "-"*99+"\n",
              "overall\tf1_score:{f1:.4f}\taccuracy:{acc:.4f}\n".format(
                  f1=metrics["overall"]["f1_score"],
                  acc=metrics["overall"]["accuracy"]),
              "confusion matrix:\n {}".format(metrics["overall"]["confusion_matrix"]))
    return metrics

def convert_to_gray(x, percentile=99):
    """
    Args:
        x: torch tensor with shape of (1, 3, H, W)
        percentile: int
    Return:
        result: shape of (1, 1, H, W)
    """
    x_2d = torch.abs(x).sum(dim=1).squeeze(0)
    v_max = np.percentile(x_2d, percentile)
    v_min = torch.min(x_2d)
    torch.clamp_((x_2d - v_min) / (v_max - v_min), 0, 1)
    return x_2d.unsqueeze(0).unsqueeze(0)

