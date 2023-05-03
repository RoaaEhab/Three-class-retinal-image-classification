import numpy as np
import os
from PIL import Image

directory = 'Chosen dataset\\Exudate_resized'
directory_resized = 'Chosen dataset\\Exudate_preprocessed'

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        # Open the image
        image = Image.open(os.path.join(directory, filename))
        # convert to np array
        image_array = np.array(image)

        # split the image into the three channels R,G,B
        red_channel = image_array[:,:,0]
        green_channel = image_array[:,:,1]
        blue_channel = image_array[:,:,2]

        # normalize the green channel using standardization
        mean = np.mean(green_channel)
        std = np.std(green_channel)
        normalized= (green_channel- mean) / std

        # rescale the image intensities to 0-255 
        scaled = 255 * (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
        image_array[:,:,1] = scaled
        preprocessed_image = Image.fromarray(image_array)

        # Save the rescaled image with a new filename
        new_filename = os.path.splitext(filename)[0] + '_processed.jpg'
        preprocessed_image.save(os.path.join(directory_resized, new_filename))

# # multiply by mask