import numpy as np
import cv2

cascade = cv2.CascadeClassifier('cascade.xml')

img = cv2.imread('pics/gamePlay2.png')



cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()