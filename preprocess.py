import cv2
import numpy as np
import os

# Set numpy print options
np.set_printoptions(linewidth=np.inf, formatter={'float': '{: 0.6f}'.format})

# Define the directory containing the images
input_directory = 'test/'
output_directory = 'txt_files/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over each file in the directory
for filename in os.listdir(input_directory):
    # Check if the file is an image
    if filename.endswith(".png"):
        input_filepath = os.path.join(input_directory, filename)
        
        # Read the image
        img = cv2.imread(input_filepath, 0)
        
        # Resize if necessary
        if img.shape != [28, 28]:
            img = cv2.resize(img, (28, 28))
        
        # Reshape the image
        img = img.reshape(28, 28, -1)
        
        # Revert and normalize the image
        img = 1.0 - img / 255.0
        
        # Convert to numpy matrix
        z = np.matrix(img)
        
        # Create the output filename with the same name but with .txt extension
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_filepath = os.path.join(output_directory, output_filename)
        
        # Open the file for writing
        with open(output_filepath, 'w') as f:
            # Write the matrix to the file
            for i in range(28):
                for j in range(28):
                    f.write(str(z[i, j]) + ' ')
                f.write('\n')
