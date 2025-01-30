import numpy as np

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

saveroot = r"F:\task9\me\mmc-amd-main2.2\600CFP.npz"
# 读取保存的数据
f = np.load(saveroot, allow_pickle=True)
pcfps = f['arr_1']

saveroot = r"F:\task9\me\mmc-amd-main2.2\600OCT.npz"
# 读取保存的数据
f = np.load(saveroot, allow_pickle=True)
pocts = f['arr_1']

predict_list = []
for i in range(len(pcfps)):
    pcfp = pcfps[i]
    poct = pocts[i]
    dcfp = class_probability_translation(pcfp)
    doct = class_probability_translation(poct)
    combined_belief = combine_beliefs(dcfp, doct)
    # combined_belief = combine_beliefs3(dcfp,doct,doct)
    print(combined_belief)
    predict_list.append(np.argmax(combined_belief))


predict_list = np.array(predict_list)
print(predict_list)

