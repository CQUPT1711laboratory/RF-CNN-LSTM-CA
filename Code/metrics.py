import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import recall_score, auc, f1_score, precision_score, hamming_loss

base_landuse = np.loadtxt('../cq21/data/cq2010_res6.txt', skiprows=6)
start1 = base_landuse[base_landuse != -9999]
end_landuse = np.loadtxt('../cq21/data/cq2020_res6.txt', skiprows=6)
end1 = end_landuse[base_landuse != -9999]
# The predicted result.
pre = np.loadtxt('../cq21/2025/result/temp1.txt', skiprows=6)
pre1 = pre[base_landuse != -9999]

misses, hits, wrongHits, falseAlarms = 0, 0, 0, 0
for x, y, z in zip(start1, pre1, end1):
    if z != x:
        # 实际上发生改变
        if y == x:
            misses += 1  # 预测没有发生改变的错误区域
        else:
            if (y == z):
                hits += 1  # 捕捉到实际的变化的正确区域
            if (y != z):
                wrongHits += 1  # 捕捉到变化，但是预测错误的区域
    else:
        # 实际上没有发生改变
        if y != x:
            # 模拟时发生改变的错误区域
            falseAlarms += 1

print("---------------All-------------")
print(misses, hits, wrongHits, falseAlarms)
print("All_FOM：", (hits * 1.0) / ((misses + hits + wrongHits + falseAlarms + 0.000001) * 1.0))
print('All_kappa:', cohen_kappa_score(pre1, end1))

# (None, ‘micro’, ‘macro’, ‘weighted’, ‘samples’)
print("accuracy_score: ", accuracy_score(end1, pre1))  # 算出精确的分数
print('recall_score: ', recall_score(end1, pre1, average='macro'))
print('precision_score: ', precision_score(end1, pre1, average='macro'))
print('f1_score: ', f1_score(end1, pre1, average='macro'))
print('hamming_loss: ', hamming_loss(end1, pre1))
