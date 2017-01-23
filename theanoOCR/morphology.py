import cv2
from matplotlib import pyplot as plt
import numpy as np
from theanoOCR import openimage as op
import scipy.fftpack

# path = os.path.join(os.path.dirname(__file__), 'Original_letters')
# inImg = cv2.imread(path+os.sep+"ha-056.jpg")
inImg = op.getImage()
img = cv2.cvtColor(inImg, cv2.COLOR_BGR2GRAY)
# global thresholding
# ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
erosion = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel,iterations = 1)
#####
sure_bg = cv2.dilate(erosion,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(erosion,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(inImg,markers)
inImg[markers == -1] = [255,0,0]

print(len(markers))
cv2.imshow("dsa",inImg)
cv2.imwrite("watershed.jpg",inImg)
cv2.waitKey()
###########
erosion_clone = np.copy(erosion)  # np.array(erosion,copy=True)
print(erosion.shape)
# close = cv2.morphologyEx(th3, cv2.MORPH_HITMISS, kernel)
# open = cv2.morphologyEx(close, cv2.MORPH_HITMISS, kernel)
_, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow("threshold",open)
# cv2.waitKey(5000)
mask = np.ones(erosion_clone.shape, dtype="uint8") * 255


count=0
for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h > 300 and w > 300:
        cv2.drawContours(mask, [contour], -1, 0, cv2.FILLED)


    # discard areas that are too small
    elif h < 35 or w < 35:
        cv2.drawContours(mask, [contour], -1, 0, cv2.FILLED)

    else:
      count += 1
      skin = np.zeros(erosion_clone.shape, dtype="uint8")
      skin=  cv2.drawContours(skin, [contour], -1,255, cv2.FILLED)
      result = np.zeros(erosion_clone.shape, np.uint8)
      result = np.where(skin == 0, result, erosion_clone)
      cv2.imwrite(str(count)+"contoured.jpg", result)
      # cv2.imshow("contoured", result)
        # cv2.drawContours(skin, contours, -1, (128,255,0),1)

        # draw rectangle around contour on original image
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

# write original image with added contours to disk
new = cv2.bitwise_and(erosion_clone, mask)

print(count)
cv2.waitKey()
# cv2.drawContours(im2, contours, -1, (0,255,100), 1)
# plot all the images and their histograms

# images = [
#           blur, 0, im2]
# titles = [
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# for i in range(1):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()
