from urllib.request import urlopen
from ssl import SSLContext, PROTOCOL_TLSv1
import numpy as n
import cv2 as c

faceDetect = c.CascadeClassifier('haarcascade_frontalface_default.xml')
personId = 1
sampleCounter = 0
ip = 'https://192.168.1.93:8080/shot.jpg'

while True:
    contxt = SSLContext(PROTOCOL_TLSv1)
    inf = urlopen(ip, context=contxt).read()
    nImg = n.array(bytearray(inf), dtype=n.uint8)
    imgFrame = c.imdecode(nImg, -1)
    frame2gray = c.cvtColor(imgFrame, c.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(frame2gray, 1.3, 5)

    for x, y, w, z in faces:
        c.rectangle(imgFrame, (x, y), (x+w, y+z), (255, 0, 0), 2)
        sampleCounter += 1
        c.imwrite('dataset/person.' + str(personId) + '.' + str(sampleCounter) + '.jpg', frame2gray[y:y+z, x:x+w])
        print(str(personId) + ':' + str(sampleCounter))
        c.imshow('frame', imgFrame)

    key = c.waitKey(32)
    if key == 27:
        break

    elif sampleCounter == 91:
        break
