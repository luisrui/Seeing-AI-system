import dlib
import cv2 as cv
import numpy as np

"""
Using the function wearmask would create picture of a man wearing a mask of N95 type or
normal mask.
The function's input is the original image and mask, and output would be a picture and a valid number:
If the valid number is True, the output picture would be on mask, vise versa. 
"""


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def coverface(img):
    p = np.random.randint(0, 3)
    colors = [(250, 206, 135), (0, 0, 0), (255, 255, 255)]
    count = 0
    detector = dlib.get_frontal_face_detector()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dets = detector(img_gray, 0)
    if len(dets) != 0:
        d = dets[0]
    else:
        count += 1
        cv.imwrite(f'./LFW_Folder/error/{count}.jpg', img)
        return
    predictor = dlib.shape_predictor('./appendix/shape_predictor_68_face_landmarks.dat')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    shape = predictor(img_gray, d)
    points = shape_to_np(shape)
    bowl_shape = np.array([points[i] for i in range(2, 15)])
    bowl_shape = np.append(bowl_shape, points[28]).reshape(14, 2)
    tmp = img.copy()
    ds = cv.fillPoly(tmp, [bowl_shape], colors[p])
    return ds


def mouth_location(img):
    detector = dlib.get_frontal_face_detector()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dets = detector(img_gray, 0)
    count = 0
    if len(dets) != 0:
        d = dets[0]
    else:
        count += 1
        cv.imwrite(f'./LFW_Folder/error/mo{count}.jpg', img)
        return -1, -1, -1, -1, -1
    predictor = dlib.shape_predictor('./appendix/shape_predictor_68_face_landmarks.dat')
    height = d.bottom() - d.top()
    width = d.right() - d.left()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    shape = predictor(img_gray, d)
    target_points = [i for i in range(2, 15)] + [28]
    x, y = [], []
    for i in target_points:
        x.append(shape.part(i).x)
        y.append(shape.part(i).y)
    y_max = (int)(max(y) + height / 4)
    y_min = (int)(min(y) - height / 4)
    x_max = (int)(max(x) + 4 * width / 7)
    x_min = (int)(min(x) - 4 * width / 7)
    size = ((x_max-x_min), (y_max-y_min))
    return x_max, x_min, y_max, y_min, size


def wearmask(img):
    x_max, x_min, y_max, y_min, size = mouth_location(img)
    p = np.random.randint(0, 2)
    if size != -1:
        if p == 0:
            mask = cv.resize(mask, size)
            mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            rows, cols, channels = mask.shape
            roi = img[y_min: y_min + rows, x_min:x_min + cols]
            for r in range(rows):
                for c in range(cols):
                    if mask_gray[r][c] == 0:
                        for i in range(3):
                            mask[r][c][i] = 255
            mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            ret, key = cv.threshold(mask_gray, 254, 255, cv.THRESH_BINARY)
            mask_inv = cv.bitwise_not(key)
            try:
                img1_bg = cv.bitwise_and(roi, roi, mask=key)
                img2_fg = cv.bitwise_and(mask, mask, mask=mask_inv)
                dst = cv.add(img1_bg, img2_fg)
                return dst, True
            except:
                return coverface(img), True
        else:
            return coverface(img), True
    else:
        return img, False
