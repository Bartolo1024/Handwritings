import numpy as np
import cv2

image = cv2.imread('testpic.jpg')
cv2.imshow('test', image)
key = cv2.waitKey(0)
if key is 27:
    cv2.destroyAllWindows()
elif key is 's':
    cv2.imwrite('wtestpic.jpg', image)
    cv2.destroyAllWindows()
