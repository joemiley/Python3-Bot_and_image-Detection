import cv2 as cv
import numpy as np
from PIL import ImageGrab

# Enable camera


# import cascade file for facial recognition
HaarCascade = cv.CascadeClassifier("cascade-large-sample.xml")

'''
    # if you want to detect any object for example eyes, use one more layer of classifier as below:
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
'''

while True:
    img = np.array(ImageGrab.grab(bbox=(0, 30, 1440, 900)))
    img = cv.cvtColor(np.array(img), cv.COLOR_BGR2RGB)
    # img = imutils.resize(img, width=1200)

    #             b   g   r
    # lowerBound = (0, 0, 0)
    # upperBound = (60, 80, 80)
    # #
    # img_binary = cv.inRange(img, lowerBound, upperBound)
    # img = cv.bitwise_and(img, img, mask= img_binary)
    # inverted
    # img = cv.bitwise_and(img, img, mask=255-img_binary)

    # Getting corners around the face
    boundingBox = HaarCascade.detectMultiScale(img, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
    # drawing bounding box around face
    for (x, y, w, h) in boundingBox:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow('box_detect', img)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
cv.destroyWindow('box_detect')

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