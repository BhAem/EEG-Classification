import math

import numpy
import scipy.io

import EEGDataset
from tsaug.visualization import plot
import matplotlib.pyplot as plt
from tsaug import AddNoise, Dropout, Drift, Reverse

# subjectfile = scipy.io.loadmat(f'./s1.mat')
# print(subjectfile)

# DataSub.shape (8, 1024, 180)
# LabSub.shape (180, 1)


# EEGData_Train = EEGDataset.getSSVEP12Intra(subject=1, train_ratio=0.8, mode='train')
# EEGData_Train, EEGLabel_Train = EEGData_Train[:]
# EEGData_Train = EEGData_Train[:, :, :, :int(256 * 0.5)]
#
# EEGData_Train = EEGData_Train.squeeze(1).contiguous()
# n, c, t = EEGData_Train.size()
# EEGData_Train = EEGData_Train.view(n, EEGData_Train.size(-1), c).numpy()
# plot(EEGData_Train[0, :, 0])
# plt.show()

# aaa = AddNoise(scale=(0.01, 0.05))
# EEGData_Train[0, :, 0] = aaa.augment(EEGData_Train[0, :, 0])
# plot(EEGData_Train[0, :, 0])
# plt.show()

# my_augmenter = (
#     Drift() @ 0.5
#     + Reverse() @ 0.5
#     + AddNoise() @ 0.5
#     + Dropout() @ 0.5
# )
#
# bbb = Dropout()  # drop out 10% of the time points (dropped out units are 1 ms, 10 ms, or 100 ms) and fill the dropped out points with zeros
# EEGData_Train[0, :, 0] = my_augmenter.augment(EEGData_Train[0, :, 0])
# plot(EEGData_Train[0, :, 0])
# plt.show()

# def ITR(C, M, P):
#     return C*(math.log2(M) + P*math.log2(P) + (1-P)*math.log2((1-P)/(M-1)))
#
#
# B = ITR(20, 6, 0.95)
# print(B)

a = [100, 90, 100, 100, 80, 90]
print(numpy.mean(a))
print(numpy.std(a))

