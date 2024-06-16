import cv2
import numpy as np
from PIL import ImageGrab
import imutils

# Enable camera


# import cascade file for facial recognition
faceCascade = cv2.CascadeClassifier("cascade.xml")

'''
    # if you want to detect any object for example eyes, use one more layer of classifier as below:
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
'''

while True:
    img = np.array(ImageGrab.grab(bbox=(0, 40, 1400, 900)))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img = imutils.resize(img, width=1200)

   # success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting corners around the face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
    # drawing bounding box around face
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)


    # # detecting eyes
    # eyes = eyeCascade.detectMultiScale(imgGray)
    # # drawing bounding box for eyes
    # for (ex, ey, ew, eh) in eyes:
    #     img = cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 3)


    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyWindow('face_detect')

########################################################################################################
#
# thresh holding
#
# # Remember -> OpenCV stores things in BGR order
# lowerBound = cv.Scalar(120, 80, 100);
# upperBound = cv.Scalar(140, 85, 110);
#
# # this gives you the mask for those in the ranges you specified,
# # but you want the inverse, so we'll add bitwise_not...
# cv.InRange(cv_im, lowerBound, upperBound, cv_rgb_thresh);
# cv.Not(cv_rgb_thresh, cv_rgb_thresh);