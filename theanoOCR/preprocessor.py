import cv2
import numpy as np
import os


def is_contour_bad(c):
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # the contour is 'bad' if it is not a rectangle
    return not len(approx) == 4


def counter_finder(inputImage,img):
    # print(type(inputImage))
    # img = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    # print(type(img))

    edged = cv2.Canny(img, 50, 100)

    # find contours in the image and initialize the mask that will be
    # used to remove the bad contours
    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(inputImage.shape[:2], dtype="uint8") * 255

    # loop over the contours
    for c in cnts:
        # if the contour is bad, draw it on the mask
        if is_contour_bad(c):
            cv2.drawContours(mask, [c], -1, 0, -1)

    # remove the contours from the image and show the resulting images
    img = cv2.bitwise_and(inputImage, inputImage, mask=mask)
    img = cv2.bitwise_and(img, img, mask=mask)
    # print(type(img))
    img = np.array(img, np.uint8)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def pre_processing():

    path = os.path.join(os.path.dirname(__file__), 'Intermidiate_Step')
    path2 = os.path.join(os.path.dirname(__file__), 'Original_input_letters')

    listing = os.listdir(path)

    listing.sort()
    for file in listing:

        inputImage = cv2.imread(path+"/"+file)

        img = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        img = counter_finder(inputImage, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)

        img = cv2.medianBlur(img, 5)
        # img = cv2.fastNlMeansDenoising(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # img = cv2.bilateralFilter(img, 9, 75, 75)
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        # img = cv2.fastNlMeansDenoising(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        done = False

        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel, iterations=4)

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded,  element)
            temp = cv2.subtract(img, temp)

            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        # skel = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel,iterations=10)
        # skel = cv2.morphologyEx(skel, cv2.MORPH_OPEN, kernel)
        #
        # skel = cv2.dilate(skel, kernel,iterations=8)
        # skel = cv2.erode(skel, kernel,iterations=6)
        #
        # skel = cv2.dilate(skel, kernel)
        skel = counter_finder(skel,skel)
        skel = counter_finder(skel, skel)
        skel = counter_finder(skel, skel)
        skel = counter_finder(skel, skel)
        skel = counter_finder(skel, skel)
        skel = counter_finder(skel, skel)
        skel = counter_finder(skel, skel)
        skel = counter_finder(skel, skel)
        skel = counter_finder(skel, skel)
        cv2.imwrite(path2+"/"+file, skel)
        # dilate, erode, erode, dilate


pre_processing()
# cv2.imshow("After1", img)