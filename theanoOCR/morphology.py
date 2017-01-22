import cv2
from matplotlib import pyplot as plt
import numpy as np
from theanoOCR import openimage as op

# path = os.path.join(os.path.dirname(__file__), 'Original_letters')
# inImg = cv2.imread(path+os.sep+"ha-056.jpg")
img = op.getImage()
# global thresholding
# ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((6,6),np.uint8)
erosion = cv2.morphologyEx(th3, cv2.MORPH_HITMISS, kernel)
# im2, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im2, contours, -1, (0,255,100), 1)
# plot all the images and their histograms
images = [
          blur, 0, erosion]
titles = [
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(1):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

# edges = cv2.Canny(img, 200, 500)
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()