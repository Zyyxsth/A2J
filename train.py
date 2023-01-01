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
TrainImgFrames = 3000
TestImgFrames = 1000
keypointsNumber = 14
cropWidth = 128
cropHeight = 128
batch_size = 32 # default：64
learning_rate = 5e-4 # default：0.00035
Weight_Decay = 1e-4 # default：1e-4
nepoch = 35
RegLossFactor = 5 # default：3
spatialFactor = 0.5 # default：0.5
RandCropShift = 5

randomseed = 0
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)

save_dir = './result'

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


def train():
    train_image_datasets = my_dataloader(FileDir="NYU_part",mode="train")
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_size,
                                                    shuffle=True, num_workers=6)

    test_image_datasets = my_dataloader(FileDir="NYU_part",mode="test")
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size,
                                                   shuffle=False, num_workers=6)

    net = model.A2J_model(num_classes = keypointsNumber)
    net = net.cuda()

    post_precess = anchor.post_process(shape=[cropHeight//16 ,cropWidth//16] ,stride=16 ,P_h=None, P_w=None)
    criterion = anchor.A2J_loss(shape=[cropHeight//16 ,cropWidth//16] ,stride=16, spatialFactor=spatialFactor ,P_h=None, P_w=None)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                        filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')
    epoch = 0
    for epoch in range(nepoch):
        net = net.train()
        train_loss_add = 0.0
        Cls_loss_add = 0.0
        Reg_loss_add = 0.0
        timer = time.time()

        # Training loop
        for i, (img, label) in enumerate(train_dataloaders):

            torch.cuda.synchronize()

            img, label = img.cuda(), label.cuda()

            heads = net(img)
            # print(regression)
            optimizer.zero_grad()

            Cls_loss, Reg_loss = criterion(heads, label)

            loss = 1 * Cls_loss + Reg_loss * RegLossFactor
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            train_loss_add = train_loss_add + (loss.item()) * len(img)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item()) * len(img)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item()) * len(img)

            # printing loss info
            if i % 10 == 0:
                print('epoch: ', epoch, ' step: ', i, 'Cls_loss ', Cls_loss.item(), 'Reg_loss ', Reg_loss.item(),
                      ' total loss ', loss.item())

        scheduler.step(epoch)

        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / TrainImgFrames
        print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))

        train_loss_add = train_loss_add / TrainImgFrames
        Cls_loss_add = Cls_loss_add / TrainImgFrames
        Reg_loss_add = Reg_loss_add / TrainImgFrames
        print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' % (train_loss_add, TrainImgFrames))
        print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' % (Cls_loss_add, TrainImgFrames))
        print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' % (Reg_loss_add, TrainImgFrames))

        Error_test = 0
        Error_train = 0
        Error_test_wrist = 0

        if (epoch % 1 == 0):
            net = net.eval()
            output = torch.FloatTensor()
            labels = torch.FloatTensor()
            outputTrain = torch.FloatTensor()

            for i, (img, label) in tqdm(enumerate(test_dataloaders)):
                with torch.no_grad():
                    img, label = img.cuda(), label.cuda()
                    heads = net(img)
                    pred_keypoints = post_precess(heads)
                    output = torch.cat([output, pred_keypoints.data.cpu()], 0)
                    labels = torch.cat([labels, label.data.cpu()], 0)

            result = output.cpu().data.numpy()
            labels = labels.cpu().data.numpy()
            Error_test = errorCompute(result, labels)
            print('epoch: ', epoch, 'Test error:', Error_test)
        # log
        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, Err_test=%.4f, lr = %.6f'
                     % (epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_lr()[0]))
    saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(Weight_Decay) + '_spatial_' + str(
        spatialFactor) + '_RegLoss_' + str(RegLossFactor)
    torch.save(net.state_dict(), saveNamePrefix + '.pth')


def errorCompute(source, target):
    assert np.shape(source) == np.shape(target), "source has different shape with target"

    outputs = source.copy()
    labels = target.copy()

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)


if __name__ == '__main__':
    train()












