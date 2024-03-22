import cv2
import numpy as np

np.set_printoptions(linewidth=np.inf, formatter={'float': '{: 0.6f}'.format})

# Load the image from the 'test' directory
img = cv2.imread('test/000000-num7.png', 0)

if img.shape != [28, 28]:
    img = cv2.resize(img, (28, 28))

img = img.reshape(28, 28, -1)

# Invert and normalize the image to the 0-1 range
img = 1.0 - img / 255.0

# Flatten the image array
img_flat = img.flatten()

# Write the flattened image array to the output file
with open('output.txt', 'w') as f:
    for pixel_value in img_flat:
        f.write(f'{pixel_value:0.6f}\n')
