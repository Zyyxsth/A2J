import os
import numpy as np
depth_train=np.load("NYU_part/depth_train.npy")
print(depth_train.shape)
label_train=np.load("NYU_part/label_train.npy")
print(label_train.shape)
depth_test=np.load("NYU_part/depth_test.npy")
print(depth_test.shape)
label_test=np.load("NYU_part/label_test.npy")
print(label_test.shape)