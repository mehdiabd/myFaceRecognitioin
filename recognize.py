__author__ = "Mehdi Abdullahi"

import cv2
import numpy as n
from ssl import SSLContext, PROTOCOL_TLSv1
from urllib.request import urlopen as url
import datetime

recognize = cv2.cv2.face.LBPHFaceRecognizer_create()
recognize.read('trainer/trainer.yml')
cascade = 'haarcascade_frontalface_default.xml'
faceClassifier = cv2.CascadeClassifier(cascade)
fontStyle = cv2.FONT_HERSHEY_SIMPLEX
webcamServerIP = 'https://192.168.1.93:8080/shot.jpg'

while True:

    now = datetime.datetime.now()
    contxt = SSLContext(PROTOCOL_TLSv1)
    inf = url(webcamServerIP, context=contxt).read()
    npImg = n.array(bytearray(inf), dtype=n.uint8)
    i = cv2.imdecode(npImg, -1)
    grayscale = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    face = faceClassifier.detectMultiScale(grayscale, 1.3, 5)
    unknownCounter = 0

    with open("personnel.log", "w") as f:

        for x, y, w, z in face:

            cv2.rectangle(i, (x-20, y-20), (x+w+20, y+z+20), (0, 255, 0), 4)
            ID, person = recognize.predict(grayscale[y:y+z, x:x+w])
            log = []

            if ID == 1:
                ID = 'Mehdi'
                log.append("{} has passed the gate on {}".format(ID, now))
            elif ID == 2:
                ID = 'Abdullah'
                log.append("{} has passed the gate on {}".format(ID, now))
            else:
                unknownCounter += 1
                ID = 'Unknown{}'.format(unknownCounter)
                log.append("{} has passed the gate on {}".format(ID, now))

            cv2.rectangle(i, (x-22, y-90), (x+w+22, y-22), (0, 255, 0), -1)
            cv2.putText(i, str(ID), (x, y-40), fontStyle, 2, (255, 255, 255), 3)

    cv2.imshow('im', i)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
