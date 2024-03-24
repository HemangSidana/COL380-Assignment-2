import cv2
import numpy as np

np.set_printoptions(linewidth=np.inf,formatter={'float': '{: 0.6f}'.format})

img = cv2.imread('test/000000-num7.png',0)
if img.shape != [28,28]:
    img2 = cv2.resize(img,(28,28))
    
img = img2.reshape(28,28,-1)

#revert the image,and normalize it to 0-1 range
img = 1.0 - img/255.0

# print(np.matrix(img))
z= np.matrix(img)
for i in range (28):
    for j in range (28):
        print(z[i,j],end=' ')
    print('')