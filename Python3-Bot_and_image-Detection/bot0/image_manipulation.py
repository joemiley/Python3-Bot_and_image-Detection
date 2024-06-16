import cv2
import numpy as np
from PIL import ImageGrab
import time

#HaarCascade = cv2.CascadeClassifier("cascade-harsh-threshold.xml")
HaarCascade = cv2.CascadeClassifier("cascade-top-harsh.xml")
#HaarCascade = cv2.CascadeClassifier("cascade-normal-colour.xml")
#HaarCascade = cv2.CascadeClassifier("cascade-topdown.xml")
#HaarCascade = cv2.CascadeClassifier("cascade-large-sample.xml")

while True:
    img = np.array(ImageGrab.grab(bbox=(0, 0, 1440, 900)))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img = cv2.multiply(img, 6)
    #
    img_Height = int(img.shape[0] * 0.2)
    img_width = int(img.shape[1] * 0.2)
    #
    img = cv2.resize(img, (img_width, img_Height), interpolation=cv2.INTER_AREA)
    img_min = np.min(img)
    img_max = np.max(img)

    # print("max value = " + str(img_max))
    # print("min value = " + str(img_min))

    img_scale = 5

    img_Height = int(img.shape[0] * img_scale)
    img_width = int(img.shape[1] * img_scale)
    #
    img = cv2.resize(img, (img_width, img_Height), interpolation=cv2.INTER_NEAREST)

    boundingBox = HaarCascade.detectMultiScale(img,
                                               scaleFactor=1.1,
                                               minNeighbors=0,
                                               flags=0,
                                               minSize=(20, 20),
                                               maxSize=(50, 50),
                                               )  # 1.3 = scale factor, 5 = minimum neighbor
    # drawing bounding box around face
    for (x, y, w, h) in boundingBox:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    img = cv2.rectangle(img, (30, 30), (80, 80), (0, 0, 255), 3)

    cv2.imshow('box_detect', img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyWindow('box_detect')


