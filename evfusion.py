import numpy as np
from lib.test_lib import test_class, show_cm

def class_probability_translation(prob_vec):
    """将神经网络输出概率向量转换为信任度分布"""
    prob_vec = np.maximum(0, prob_vec)  #实现ReLU的功能，去除负值对预测的影响
    d = prob_vec / prob_vec.sum()   #归一化

    return d

def combine_beliefs(a,b):
    """证据理论融合，将两个证据的信任度分布进行合并"""
    # combined_belief = np.zeros((c,c))  # 初始化合并后的信任度分布
    combined_belief = np.dot(a[:, np.newaxis], b[np.newaxis, :])
    # 提取有效结果
    belief_diag = np.diag(combined_belief)
    diagsum = belief_diag.sum()
    if diagsum>0:   #如果全为零的话就不做处理了
        belief_diag = belief_diag / belief_diag.sum()   #分子为1-k表示有效集合概率

    return belief_diag

def combine_beliefs3(a,b,c):
    """证据理论融合，将三个证据的信任度分布进行合并"""
    l = len(a)
    # combined_belief = np.zeros((c,c))  # 初始化合并后的信任度分布
    combined_belief = np.dot(a[:, np.newaxis], b[np.newaxis, :])
    combined_belief = np.dot(combined_belief[:, :, np.newaxis], c[np.newaxis, np.newaxis, :])
    combined_belief = np.squeeze(combined_belief)
    # 提取有效结果
    belief_diag = np.zeros(l)
    for i in range(l):
        belief_diag[i] = combined_belief[i,i,i]
    belief_diag = belief_diag / belief_diag.sum()   #分子为1-k表示有效集合概率
    return belief_diag


class_name = ['Normal', 'CNV', 'DR', 'AMD', 'CSC', 'RVO', 'Others']

#处理cfp-mask-data
saveroot = r"output\cfp-mask-data.npz"
f = np.load(saveroot, allow_pickle=True)
cfp_mask_pre = f['arr_1']
gt = f['arr_0']
# show_cm(cfp_mask_pre, gt, class_name, 'output/cfp_mask ')

#处理cfp-graph-data.
saveroot = r"output\cfp-graph-data.npz"
f = np.load(saveroot, allow_pickle=True)
cfp_graph_pre = f['arr_1']
gt = f['arr_0']
# show_cm(cfp_graph_pre, gt, class_name, 'output/cfp_graph ')

#处理oct-mask-data
saveroot = r"output\oct-mask-data.npz"
f = np.load(saveroot, allow_pickle=True)
oct_mask_pre = f['arr_2']
oct_cnn_pre = f['arr_1']
gt = f['arr_0']
# show_cm(oct_cnn_pre, gt, class_name, 'output/oct_cnn ')
# show_cm(oct_mask_pre, gt, class_name, 'output/oct_mask ')

#处理oct-graph-data.
saveroot = r"output\oct-graph-data.npz"
f = np.load(saveroot, allow_pickle=True)
oct_graph_pre = f['arr_1']
gt = f['arr_0']
# show_cm(oct_graph_pre, gt, class_name, 'output/oct_graph ')

# cfp证据融合
cps = []
for i in range(len(gt)):
    cmp = cfp_mask_pre[i]
    cgp = cfp_graph_pre[i]
    dcmp = class_probability_translation(cmp)
    dcgp = class_probability_translation(cgp)
    cp = combine_beliefs(dcmp, dcgp)
    cps.append(cp)
    # print(cp)
cps = np.array(cps)
test_class(class_name,gt,cps,path_experiment='output/cfp_')

#oct证据融合
ops = []
for i in range(len(gt)):
    ocp = oct_cnn_pre[i]
    omp = oct_mask_pre[i]
    ogp = oct_graph_pre[i]
    docp = class_probability_translation(ocp)
    dcgp = class_probability_translation(omp)
    dogp = class_probability_translation(ogp)
    op = combine_beliefs3(docp,dcgp,dogp)
    ops.append(op)
    # print(op)
ops = np.array(ops)
test_class(class_name,gt,ops,path_experiment='output/oct_')

# 多模态证据融合
ps = []
for i in range(len(gt)):
    cp = cps[i]
    op = ops[i]
    p = combine_beliefs(cp, op)
    ps.append(p)
    # print(cp)
ps = np.array(ps)
test_class(class_name,gt,ps,path_experiment='output/all_')

