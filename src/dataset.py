import cv2
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from src import random_erasing
import logging
import time
import datetime
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

u0 = 64
v0 = 64

# DataHyperParms
TrainImgFrames = 3000
TestImgFrames = 1000
keypointsNumber = 14
cropWidth = 128  # 176
cropHeight = 128  # 176
batch_size = 64
learning_rate = 0.00035
Weight_Decay = 1e-4
nepoch = 35
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 1
RandRotate = 180
RandScale = (1.0, 0.5)
# xy_thres = 110
# depth_thres = 150

randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)

save_dir = './result/NYU_batch_64_12345'

try:
    os.makedirs(save_dir)
except OSError:
    pass



joint_id_to_name = {
    0: 'pinky tip',
    1: 'pinky mid',
    2: 'ring tip',
    3: 'ring mid',
    4: 'middle tip',
    5: 'middle mid',
    6: 'index tip',
    7: 'index mid',
    8: 'thumb tip',
    9: 'thumb mid',
    10: 'thumb root',
    11: 'wrist back',
    12: 'wrist',
    13: 'palm',
}


def transform(img, label, matrix):
    '''
    img: [H, W]  label, [N,2]
    '''
    img_out = cv2.warpAffine(img ,matrix ,(cropWidth ,cropHeight))
    label_out = np.ones((keypointsNumber, 3))
    label_out[: ,:2] = label[: ,:2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, label_out


def dataPreprocess(data, label, augment=True):

    # if augment:
    #     imgResize, label_xy = transform(imgResize, label_xy, matrix)  ## rotation, scale
    if random.random()<0.5:
        data = data[:, ::-1] # [H,W]
        label[1] = -label[1]

    return data, label


class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, FileDir, mode="train", augment=True):
        self.FileDir = FileDir
        self.depths = np.load(FileDir+"/depth_{}.npy".format(mode)) # [N,W,H]
        # self.depths = np.transpose(self.depths,(0,2,1)) # [N,H,W]
        self.labels = np.load(FileDir+"/label_{}.npy".format(mode))
        self.mode = mode
        self.augment = augment
        # self.randomErase = random_erasing.RandomErasing(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0])

    def __getitem__(self, index):

        depth = self.depths[index, :, :] # [128,128]
        label = self.labels[index, :] # [42]
        label = label.reshape((14,3))
        # data, label = dataPreprocess(depth, label, self.augment)

        depth = depth[None] # [1,128,128]
        # if self.augment:
        #     depth = self.randomErase(depth)
        return torch.Tensor(depth), torch.Tensor(label)

    def __len__(self):
        return self.depths.shape[0]


















