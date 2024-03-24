import os
import cv2
import numpy as np

# Set print options
np.set_printoptions(linewidth=np.inf, formatter={'float': '{: 0.6f}'.format})

# Function to process an image and save float values to a text file
def process_image(image_path, output_folder, counter):
    # Read the image
    img = cv2.imread(image_path, 0)
    
    # Resize the image if it's not 28x28
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))
    
    # Reshape the image
    img = img.reshape(28, 28, -1)
    
    # Revert the image and normalize it to the 0-1 range
    img = 1.0 - img / 255.0
    
    # Convert to a NumPy matrix
    z = np.matrix(img)
    
    # Create text file path
    filename = str(counter) + ".txt"
    output_path = os.path.join(output_folder, filename)
    
    # Save float values to text file
    with open(output_path, 'w') as file:
        for i in range(28):
            for j in range(28):
                file.write('{:0.6f} '.format(z[i, j]))
            file.write('\n')

# Folder containing images
input_folder = 'test'

# Create output folder if it doesn't exist
output_folder = 'image_float'
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
counter=1
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder,counter)
        counter+=1
