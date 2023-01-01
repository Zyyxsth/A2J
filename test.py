import cv2
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
from src import model as model
from src import anchor as anchor
from src.dataset import my_dataloader
from tqdm import tqdm
# import random_erasing
import logging
import time
import datetime
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DataHyperParms
keypointsNumber = 14
cropWidth = 128
cropHeight = 128
batch_size = 64

randomseed = 0
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)

model_dir = 'result/NYU.pth'


def test():
    test_image_datasets = my_dataloader(FileDir="NYU_part", mode="test")
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size,
                                                   shuffle=False, num_workers=6)

    net = model.A2J_model(num_classes=keypointsNumber)
    net.load_state_dict(torch.load(model_dir))
    net = net.cuda()
    net.eval()

    post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)

    output = torch.FloatTensor()
    labels = torch.FloatTensor()
    torch.cuda.synchronize()
    for i, (img, label) in tqdm(enumerate(test_dataloaders)):
        with torch.no_grad():
            img, label = img.cuda(), label.cuda()
            heads = net(img)
            pred_keypoints = post_precess(heads, voting=False)
            output = torch.cat([output, pred_keypoints.data.cpu()], 0)
            labels = torch.cat([labels, label.data.cpu()], 0)

    torch.cuda.synchronize()

    result = output.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    error = errorCompute(result, labels)
    print('Error:', error)


def errorCompute(source, target):
    assert np.shape(source) == np.shape(target), "source has different shape with target"

    outputs = source.copy()
    labels = target.copy()

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)


if __name__ == '__main__':
    test()

