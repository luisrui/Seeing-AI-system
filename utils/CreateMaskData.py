from Wearmask import wearmask, coverface
import cv2 as cv
import os
import numpy as np


def CreateData():
    mask = cv.imread('./imgs/masks/N95.png')
    path = './data/archive/lfw-deepfunneled'
    for folder in os.listdir(path):
        folder_path = os.path.join(path+'/' + folder)
        os.mkdir(f'./data/archive/mask/{folder}')
        for img_name in os.listdir(folder_path):
            img = cv.imread(folder_path + '/' + img_name)
            mask_img, valid = wearmask(img, mask)
            if valid:
                cv.imwrite(
                    f'./data/archive/mask/{folder}/{img_name}', mask_img)
