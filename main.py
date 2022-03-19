import cv2 as cv
import numpy as np
import matplotlib as plt
SAMPLES = 800
print(cv.__version__)
img = cv.imread('img1.jpg')


imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(imgGray, (3,3), 0)

#Canny Edge Detection
edgeme=cv.Canny(image=img_blur, threshold1=100, threshold2=200)

#SOBEL edges
cptsobel = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)

# #Create default parametrization LSD
# lsd = cv.createLineSegmentDetector(0)
# #Detect lines in the image
# lines = lsd.detect(edgeme)[0]
# #Draw detected lines in the image
# drawn_img = lsd.drawSegments(edgeme,lines)
# newlines = np.asarray(lines)
# sample = np.random.permutation(np.arange(newlines.shape[0]))[:SAMPLES]
# print(newlines.shape)
# stratified = np.empty([0,0])
# for i,each in enumerate(sample):
#     stratified = np.append(stratified, newlines[each])
# stratified = np.reshape(stratified, [SAMPLES,1,4])
# print(stratified.shape)
# stratified = stratified.astype('float32')
# drawn_img = lsd.drawSegments(edgeme, stratified)
# cv.imshow('',drawn_img)
# cv.waitKey(0)
hough = np.empty([1,1,4])
hough = cv.HoughLinesP(edgeme, 1, np.pi/180, 80, minLineLength=0, maxLineGap=10 )
for line in hough:
    x1,y1,x2,y2 = line[0]
    cv.line(edgeme, (x1,y1),(x2,y2), (255,255,0),1)
cv.imshow('',edgeme)
cv.waitKey(0)
