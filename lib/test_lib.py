import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#预测程序只保留预测结果，可视化及计算准确度在最后证据融合里做
#这里设置单独的显示函数
def test_class(class_name,ground_truth,prediction,path_experiment):#训练函数
    prediction = np.argmax(prediction, axis=1)
    file_perf = open(path_experiment + 'performances.txt', 'w')
    # Sensitivity, Specificity and F1 per class
    print('class acc, sen, spe, pre, miou, f1')
    file_perf.write('class acc, sen, spe, pre, miou, f1' + '\n')
    n_classes = len(class_name)
    altn = 0
    alfp = 0
    alfn = 0
    altp = 0
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
        altn = altn + tn
        alfp = alfp + fp
        alfn = alfn + fn
        altp = altp + tp
    aacc = float(altp + altn) / (altn + alfp + alfn + altp)
    asen = float(altp) / (alfn + altp)
    aspe = float(altn) / (altn + alfp)
    apre = float(altp) / (altp + alfp)
    amiou = float(altp) / (altp + alfp + alfn)
    af1 = 2 * apre * asen / (apre + asen)
    print('mean_of_all', '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (aacc, asen, aspe, apre, amiou, af1))
    file_perf.write('mean_of_all' + '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (aacc, asen, aspe, apre, amiou, af1) + '\n')
    file_perf.close()

def show_cm(prediction, ground_truth, class_name, path_experiment):
    prediction = np.argmax(prediction, axis=1)
    title = 'Confusion matrix'
    cm = confusion_matrix(ground_truth, prediction)
    cmap = plt.cm.Blues
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
    test_class(class_name, ground_truth, prediction, path_experiment)
    print("done")