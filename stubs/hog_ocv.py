import cv2
hog = cv2.HOGDescriptor()
im = cv2.imread('robot.jpg')
h = hog.compute(im)

cv2.imshow('image',h)
cv2.waitKey(0)
cv2.destroyAllWindows()