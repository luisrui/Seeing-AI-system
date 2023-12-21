import pandas as pd
import cv2 as cv
from LocateMaskFace import CropMaskFace
from Wearmask import wearmask

mask = cv.imread("./mask_template/N95.png")
train = pd.read_csv("train.csv")

import os

trainlist = []
for train_dir, label in zip(train.img_train, train.label_train):
    path = os.path.join("./LFW_Folder/maskpics/" + train_dir[:-9] + "/" + train_dir)
    img = cv.imread(path)
    if img is None:
        path = os.path.join(
            "./LFW_Folder/lfw_funneled/" + train_dir[:-9] + "/" + train_dir
        )
        img = cv.imread(path)
        mask_img, val = wearmask(img, mask)
        if val:
            mask_crop, valid = CropMaskFace(mask_img)
            if valid:
                filename = "train/" + train_dir
                cv.imwrite(f"./train/{train_dir}", mask_crop)
                trainlist.append([train_dir, label])
    else:
        mask_crop, valid = CropMaskFace(img)
        if valid:
            filename = "train/" + train_dir
            cv.imwrite(f"./train/{train_dir}", mask_crop)
            trainlist.append([train_dir, label])

trainlist = pd.DataFrame(trainlist)
trainlist.columns = ["img_train", "label_train"]
trainlist.to_csv("mask_train.csv")


testlist = []
test = pd.read_csv("test.csv")
for test_dir, label in zip(test.img_test, test.label_test):
    path = os.path.join("./LFW_Folder/maskpics/" + test_dir[:-9] + "/" + test_dir)
    img = cv.imread(path)
    if img is None:
        path = os.path.join(
            "./LFW_Folder/lfw_funneled/" + test_dir[:-9] + "/" + test_dir
        )
        img = cv.imread(path)
        mask_img, val = wearmask(img, mask)
        if val:
            mask_crop, valid = CropMaskFace(mask_img)
            if valid:
                filename = "test/" + test_dir
                cv.imwrite(f"./test/{test_dir}", mask_crop)
                testlist.append([test_dir, label])
    else:
        mask_crop, valid = CropMaskFace(img)
        if valid:
            filename = "test/" + test_dir
            cv.imwrite(f"./test/{test_dir}", mask_crop)
            testlist.append([test_dir, label])


vallist = []
val = pd.read_csv("val.csv")
for val_dir, label in zip(val.img_val, val.label_val):
    path = os.path.join("./LFW_Folder/maskpics/" + val_dir[:-9] + "/" + val_dir)
    img = cv.imread(path)
    if img is None:
        path = os.path.join("./LFW_Folder/lfw_funneled/" + val_dir[:-9] + "/" + val_dir)
        img = cv.imread(path)
        mask_img, val = wearmask(img, mask)
        if val:
            mask_crop, valid = CropMaskFace(mask_img)
            if valid:
                filename = "val/" + val_dir
                cv.imwrite(f"./val/{val_dir}", mask_crop)
                vallist.append([filename, label])
    else:
        mask_crop, valid = CropMaskFace(img)
        if valid:
            filename = "val/" + val_dir
            cv.imwrite(f"./val/{val_dir}", mask_crop)
            vallist.append([filename, label])
