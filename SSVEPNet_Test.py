# Designer:Pan YuDong
# Coder:God's hand
# Time:2021/11/26 19:52
import os
import torch
import argparse
import sys
from pathlib import Path

import Constraint
import LossFunction
import MLP
import EEGDataset
import Ploter
# import SSVEPNet
import SSVEPNet2
import Classifier_Trainer

# # huawei
# import moxing as mox
# local_data_url = '/cache/data/Dial/'
# local_train_url = '/cache/Result/DatasetA/SSVEPNet/'


# 1、Define parameters of eeg
'''                    
---------------------------------------------Intra-subject Experiments ---------------------------------
                        epochs    bz     lr   lr_scheduler    ws      Fs    Nt   Nc   Nh   Ns     wd
    DatasetA(1S/0.5S):  500      30    0.01      Y           1/0.5   256   1024  8   180  10  0.0003
    DatasetB(1S/0.5S):  500      16    0.01      Y           1/0.5   250   1000  8    80  10  0.0003
---------------------------------------------Inter-subject Experiments ---------------------------------
                        epochs     bz          lr       lr_scheduler  ws      Fs     Nt    Nc   Nh      wd        Kf
    DatasetA(1S/0.5S):  500/100   64/30    0.001/0.01     N/Y        1/0.5    256   1024   8    180   0/0.0001   1/5
    DatasetB(1S/0.5S):  500/100   64/30    0.001/0.01     N/Y        1/0.5    250   1000   8     80   0/0.0003   1/5 
'''

'''
公共数据集：
    epochs：500，bz：30，lr：0.01，Nh：180，Fs：256，Nt：1024，Nf：12，Ns：10
整合的自制数据集：
    epochs：200，bz：10，lr：0.01，Nh：52，Fs：2000，Nt：2000，Nf：4，Ns：1
2000-2000：
    epochs：200，bz：2，lr：0.001，Nh：8，Fs：2000，Nt：2000，Nf：4，Ns：6
1000-256：
    epochs：200，bz：2，lr：0.01，Nh：16，Fs：256，Nt：1000，Nf：4，Ns：6
500-256：
    epochs：200，bz：2，lr：0.01，Nh：16，Fs：256，Nt：500，Nf：4，Ns：12
1000-256-SSEVPNet：
    epochs：150，bz：2，lr：0.0001，Nh：16，Fs：256，Nt：1000，Nf：4，Ns：6
'''

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500, help="number of epochs") # 500 200 200 200 150
parser.add_argument('--bz', type=int, default=30, help="number of batch") # 30 10 2 2 2
parser.add_argument('--lr', type=float, default=0.01, help="learning rate") # 0.01 0.01 0.001 0.01 0.0001
parser.add_argument('--ws', type=float, default=0.5, help="window size of ssvep")
parser.add_argument('--Kf', type=int, default=1, help="k-fold cross validation")
parser.add_argument('--Nh', type=int, default=180, help="number of trial")  # 180 52 8 16 16
parser.add_argument('--Nc', type=int, default=8, help="number of channel")
parser.add_argument('--Fs', type=int, default=256, help="frequency of sample") # 256 2000 2000 256 256
parser.add_argument('--Nt', type=int, default=1024, help="number of sample") # 1024 2000 2000 1000 1000
parser.add_argument('--Nf', type=int, default=12, help="number of stimulus") # 12 4 4 4 4
parser.add_argument('--Ns', type=int, default=10, help="number of subjects") # 10 1 6 6 6
parser.add_argument('--wd', type=int, default=0.0003, help="weight decay")
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--seed', type=int, default=2023, help='seed number')
opt = parser.parse_args()
devices = "cuda" if torch.cuda.is_available() else "cpu"

import random
import numpy as np

os.environ['PYTHONHASHSEED'] = str(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# # huawei
# if not os.path.exists(local_data_url):
#     os.makedirs(local_data_url)
# if not os.path.exists(local_train_url):
#     os.makedirs(local_train_url)
# mox.file.copy_parallel(opt.data_url, local_data_url)


# 2、Start Training
best_acc_list = []
final_acc_list = []
for fold_num in range(opt.Kf):
    best_valid_acc_list = []
    final_valid_acc_list = []
    print(f"Training for K_Fold {fold_num + 1}")
    for testSubject in range(1, opt.Ns + 1):
        print(f"Training on subject {testSubject}")
        # **************************************** #
        '''12-class SSVEP Dataset'''
        # -----------Intra-Subject Experiments--------------
        # EEGData_Train = EEGDataset.getSSVEP12Intra(subject=testSubject, KFold=fold_num, n_splits=opt.Kf,
        #                                           mode='test')
        # EEGData_Test = EEGDataset.getSSVEP12Intra(subject=testSubject, KFold=fold_num, n_splits=opt.Kf,
        #                                          mode='train')

        EEGData_Train = EEGDataset.getSSVEP12Intra(subject=testSubject, train_ratio=0.8, mode='train', opt=opt)
        EEGData_Test = EEGDataset.getSSVEP12Intra(subject=testSubject, train_ratio=0.8, mode='test', opt=opt)

        # -----------Inter-Subject Experiments--------------
        # EEGData_Train = EEGDataset.getSSVEP12Inter(subject=testSubject, mode='train')
        # EEGData_Test = EEGDataset.getSSVEP12Inter(subject=testSubject, mode='test')

        EEGData_Train, EEGData_Train_Aug, EEGLabel_Train = EEGData_Train[:]
        # EEGData_Train, EEGLabel_Train = EEGData_Train[:]
        EEGData_Train = EEGData_Train[:, :, :, :int(opt.Fs * opt.ws)]
        EEGData_Train_Aug = EEGData_Train_Aug[:, :, :, :int(opt.Fs * opt.ws)]
        # print("EEGData_Train.shape", EEGData_Train.shape)
        # print("EEGLabel_Train.shape", EEGLabel_Train.shape)
        EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGData_Train_Aug, EEGLabel_Train)
        # EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGLabel_Train)

        EEGData_Test, EEGData_Test_Aug, EEGLabel_Test = EEGData_Test[:]
        # EEGData_Test, EEGLabel_Test = EEGData_Test[:]
        EEGData_Test = EEGData_Test[:, :, :, :int(opt.Fs * opt.ws)]
        # print("EEGData_Test.shape", EEGData_Test.shape)
        # print("EEGLabel_Test.shape", EEGLabel_Test.shape)
        EEGData_Test = torch.utils.data.TensorDataset(EEGData_Test, EEGLabel_Test)

        # Create DataLoader for the Dataset
        train_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=opt.bz, shuffle=True,
                                           drop_last=False, num_workers=0, pin_memory=True)
        valid_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=opt.bz, shuffle=False,
                                           drop_last=False, num_workers=0, pin_memory=True)

        # Define Network
        net = SSVEPNet2.ESNet(opt.Nc, int(opt.Fs * opt.ws), opt.Nf)
        # net = Constraint.Spectral_Normalization(net)
        net = net.to(devices)

        mlp = MLP.myMLP()
        mlp = mlp.to(devices)

        criterion = torch.nn.CrossEntropyLoss()
        # criterion = LossFunction.CELoss_Marginal_Smooth(opt.Nf, stimulus_type='4')
        valid_acc = Classifier_Trainer.train_on_batch(opt.epochs, train_dataloader, valid_dataloader, opt.lr, criterion,
                                                      net, mlp, devices, wd=opt.wd, lr_jitter=True)
        final_valid_acc_list.append(valid_acc)

    final_acc_list.append(final_valid_acc_list)


# 3、Plot Result
Ploter.plot_save_Result(final_acc_list, model_name='SSVEPNet', dataset='DatasetA', UD=0, ratio=1, win_size=str(opt.ws),
                        text=True)
# # huawei
# mox.file.copy_parallel(local_train_url, opt.train_url)
