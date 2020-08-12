import cv2
import numpy as np

img = cv2.imread("naitik.png")

imgcanny = cv2.Canny(img,100,200)


cv2.imshow("canny",imgcanny)


cv2.waitKey(0)