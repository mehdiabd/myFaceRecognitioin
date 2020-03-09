__author__ = "Mehdi Abdullahi"

import cv2, os
import numpy as n
from PIL import Image

recognize = cv2.cv2.face.LBPHFaceRecognizer_create()
detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def getImgLabel(path):

    for p in os.listdir(path):
        imgPath = [os.path.join(path, p)]

    samples = []
    IDs = []

    for path in imgPath:

        pilImg = Image.open(imgPath).convert('L')
        npImg = n.array(pilImg, 'uint8')
        imgId = int(os.path.split(path)[-1].split('.')[1])
        face = detect.detectMultiScale(npImg)

        for x, y, w, z in face:
            samples.append(npImg[y:y+z, x:x+w])
            IDs.append(imgId)

    return samples, IDs


face, IDs = getImgLabel('dataset')
recognize.train(face, n.array(IDs))
recognize.write('trainer/trainer.yml')
