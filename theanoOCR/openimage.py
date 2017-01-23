import cv2

import os


def getImage():
    path = os.path.join(os.path.dirname(__file__), 'Original_letters')
    inImg = cv2.imread(path + os.sep + "word2.jpg")



    return inImg
