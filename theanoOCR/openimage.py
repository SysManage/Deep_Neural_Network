import os

import cv2


def getImage():
    path = os.path.join(os.path.dirname(__file__), 'Original_letters')
    inImg = cv2.imread(path + os.sep + "ha-056.jpg")

    return inImg
