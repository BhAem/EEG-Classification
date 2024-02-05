# Designer:Pan YuDong
# Coder:God's hand
# Time:2021/10/6 22:47
import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io
import os
# # huawei
# os.system("pip install tsaug -i http://repo.myhuaweicloud.com/repository/pypi/simple/")
from tsaug import Dropout

# # huawei
# local_data_url = '/cache/data/Dial/'
# local_train_url = '/cache/Result/DatasetA/SSVEPNet/'


class getSSVEP12Inter(Dataset):
    def __init__(self, subject=1, mode="train"):
        self.Nh = 180
        self.Nc = 8
        self.Nt = 1024
        self.Nf = 12
        self.Fs = 256
        self.eeg_raw_data, self.eeg_raw_data_aug = self.read_EEGData()
        # self.eeg_raw_data = self.read_EEGData()
        self.label_raw_data = self.read_EEGLabel()
        if mode == 'train':
            self.eeg_data = torch.cat(
                (self.eeg_raw_data[0:(subject - 1) * self.Nh], self.eeg_raw_data[subject * self.Nh:]), dim=0)
            self.eeg_data_aug = torch.cat(
                (self.eeg_raw_data_aug[0:(subject - 1) * self.Nh], self.eeg_raw_data_aug[subject * self.Nh:]), dim=0)
            self.label_data = torch.cat(
                (self.label_raw_data[0:(subject - 1) * self.Nh:, :], self.label_raw_data[subject * self.Nh:, :]), dim=0)

        if mode == 'test':
            self.eeg_data = self.eeg_raw_data[(subject - 1) * self.Nh:subject * self.Nh]
            self.eeg_data_aug = self.eeg_raw_data_aug[(subject - 1) * self.Nh:subject * self.Nh]
            self.label_data = self.label_raw_data[(subject - 1) * self.Nh:subject * self.Nh]

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.eeg_data_aug[index], self.label_data[index]
        # return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # get the single subject data
    def get_DataSub(self, index):
        # load file into dict
        # # huawei
        subjectfile = scipy.io.loadmat(f'./data/Dial/DataSub_{index}.mat')
        # subjectfile = scipy.io.loadmat(f'/cache/data/Dial/DataSub_{index}.mat')
        # extract numpy from dict
        samples = subjectfile['Data']

        samples_aug = samples.copy()
        # c, t, n = samples_aug.shape[0], samples_aug.shape[1], samples_aug.shape[2]
        samples_aug = samples_aug.transpose((2, 1, 0))
        my_augmenter = (
            Dropout()
        )
        samples_aug = my_augmenter.augment(samples_aug)
        # n, t, c = samples_aug.shape[0], samples_aug.shape[1], samples_aug.shape[2]
        samples_aug = samples_aug.transpose((2, 1, 0))
        samples_aug = samples_aug.swapaxes(1, 2)
        eeg_data_aug = torch.from_numpy(samples_aug.swapaxes(0, 1))  # (8, 180, 1024) -> (180, 8, 1024)
        eeg_data_aug = eeg_data_aug.reshape(-1, 1, self.Nc, self.Nt)  # (180, 1, 8, 1024)

        # (num_trial, sample_point, num_trial) => (num_trial, num_channels, sample_point)
        eeg_data = samples.swapaxes(1, 2)
        eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1))
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)
        print(eeg_data.shape)
        return eeg_data, eeg_data_aug
        # return eeg_data

    def read_EEGData(self):
        eeg_data, eeg_data_aug = self.get_DataSub(1)
        for i in range(1, 10):
            single_subject_eeg_data, single_subject_eeg_data_aug = self.get_DataSub(i + 1)
            eeg_data = torch.cat((eeg_data, single_subject_eeg_data), dim=0)
            eeg_data_aug = torch.cat((eeg_data_aug, single_subject_eeg_data_aug), dim=0)
        return eeg_data, eeg_data_aug

    # get the single label data
    def get_DataLabel(self, index):
        # load file into dict
        # # huawei
        labelfile = scipy.io.loadmat(f'./data/Dial/LabSub_{index}.mat')
        # labelfile = scipy.io.loadmat(f'/cache/data/Dial/LabSub_{index}.mat')
        # extract numpy from dict
        labels = labelfile['Label']
        label_data = torch.from_numpy(labels)
        print(label_data.shape)
        return label_data - 1

    def read_EEGLabel(self):
        label_data = self.get_DataLabel(1)
        for i in range(1, 10):
            single_subject_label_data = self.get_DataLabel(i)
            label_data = torch.cat((label_data, single_subject_label_data), dim=0)
        return label_data


class getSSVEP12Intra(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train", opt=None):
        super(getSSVEP12Intra, self).__init__()
        self.Nh = opt.Nh  # number of trials
        self.Nc = opt.Nc  # number of channels
        self.Nt = opt.Nt  # number of time points
        self.Nf = opt.Nf  # number of target frequency
        self.Fs = opt.Fs  # Sample Frequency
        self.subject = subject  # current subject
        self.eeg_data, self.eeg_data_aug = self.get_DataSub()
        # self.eeg_data = self.get_DataSub()
        self.label_data = self.get_DataLabel()
        self.num_trial = self.Nh // self.Nf  # number of trials of each frequency
        self.train_idx = []
        self.test_idx = []
        if KFold is not None:
            fold_trial = self.num_trial // n_splits  # number of trials in each fold
            self.valid_trial_idx = [i for i in range(KFold * fold_trial, (KFold + 1) * fold_trial)]

        for i in range(0, self.Nh, self.Nh // self.Nf):
            for j in range(self.Nh // self.Nf):
                if n_splits == 2 and j == self.num_trial - 1:
                    continue  # if K = 2, discard the last trial of each category
                if KFold is not None:  # K-Fold Cross Validation
                    if j not in self.valid_trial_idx:
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)
                else:  # Split Ratio Validation
                    if j < int(self.num_trial * train_ratio):
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)

        self.eeg_data_train = self.eeg_data[self.train_idx]
        self.eeg_data_train_aug = self.eeg_data_aug[self.train_idx]
        self.label_data_train = self.label_data[self.train_idx]
        self.eeg_data_test = self.eeg_data[self.test_idx]
        self.label_data_test = self.label_data[self.test_idx]

        if mode == 'train':
            self.eeg_data = self.eeg_data_train
            self.label_data = self.label_data_train
        elif mode == 'test':
            self.eeg_data = self.eeg_data_test
            self.label_data = self.label_data_test

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.eeg_data_train_aug[index], self.label_data[index]
        # return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # get the single subject data
    def get_DataSub(self):
        # # huawei
        # subjectfile = np.load(f'./our_data/8channel-data5/datasub_{self.subject}.npy')
        # subjectfile = subjectfile.transpose((1, 2, 0))
        # samples = subjectfile

        subjectfile = scipy.io.loadmat(f'./data/Dial/DataSub_{self.subject}.mat')
        # subjectfile = scipy.io.loadmat(f'/cache/data/Dial/DataSub_{self.subject}.mat')
        samples = subjectfile['Data']  # (8, 1024, 180)

        samples_aug = samples.copy()
        # c, t, n = samples_aug.shape[0], samples_aug.shape[1], samples_aug.shape[2]
        samples_aug = samples_aug.transpose((2, 1, 0))
        my_augmenter = (
                Dropout(seed=2023)
        )
        samples_aug = my_augmenter.augment(samples_aug)
        # n, t, c = samples_aug.shape[0], samples_aug.shape[1], samples_aug.shape[2]
        samples_aug = samples_aug.transpose((2, 1, 0))
        samples_aug = samples_aug.swapaxes(1, 2)
        eeg_data_aug = torch.from_numpy(samples_aug.swapaxes(0, 1))  # (8, 180, 1024) -> (180, 8, 1024)
        eeg_data_aug = eeg_data_aug.reshape(-1, 1, self.Nc, self.Nt)  # (180, 1, 8, 1024)

        eeg_data = samples.swapaxes(1, 2)  # (8, 1024, 180) -> (8, 180, 1024)
        eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1))  # (8, 180, 1024) -> (180, 8, 1024)
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)  # (180, 1, 8, 1024)
        print(eeg_data.shape)
        return eeg_data, eeg_data_aug
        # return eeg_data


    # get the single label data
    def get_DataLabel(self):
        # # huawei
        # labelfileEEGlabel = np.load(f'./our_data/8channel-data5/labelsub_{self.subject}.npy')
        # labels = labelfileEEGlabel

        labelfile = scipy.io.loadmat(f'./data/Dial/LabSub_{self.subject}.mat')
        # labelfile = scipy.io.loadmat(f'/cache/data/Dial/LabSub_{self.subject}.mat')
        labels = labelfile['Label']
        label_data = torch.from_numpy(labels)
        print(label_data.shape)
        return label_data - 1
