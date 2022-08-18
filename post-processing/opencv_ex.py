import cv2
from matplotlib import pyplot as plt
import numpy as np


def cvplot(cvimg):
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(cvimg)


img = cv2.imread('mykey.jpg')
cv2.imshow('img', img)
cv2.waitKey(0)
