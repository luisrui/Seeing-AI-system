import cv2 as cv
import dlib


def crop(face, sizex, sizey):
    '''
    params:{
        face: input a face image with RGB three channels and OpenCV form
        sizex, sizey: resize the face with new params
    }
    '''
    detector = dlib.get_frontal_face_detector()
    face_gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
    dets = detector(face_gray, 0)
    if len(dets) != 0:
        d = dets[0]
        top = d.top() - 50 if d.top()-50 > 0 else 0
        bottom = d.bottom() + 20 if d.bottom() + \
            20 < face_gray.shape[0] else face_gray.shape[0]
        left = d.left() - 20 if d.left() - 20 > 0 else 0
        right = d.right() + 20 if d.right() + \
            20 < face_gray.shape[1] else face_gray.shape[1]
        face_crop = face[top:d.bottom(), left:d.right()]
        face_crop_resize = cv.resize(face_crop, dsize=(
            sizex, sizey), fx=1, fy=1, interpolation=cv.INTER_LINEAR)
        return face_crop_resize
    else:
        return face
