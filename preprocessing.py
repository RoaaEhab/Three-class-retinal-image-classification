import numpy as np
import os
from PIL import Image


directory_drusen = 'Drusen'
directory_drusen_processed = 'Processed folders\Drusen_processed'

directory_exudate = 'Exudate'
directory_exudate_processed = 'Processed folders\Exudate_processed'

directory_normal = 'Normal'
directory_normal_processed = 'Processed folders\\Normal_processed'


# Define the desired size of the resized image
new_size = (224, 224)

# preprocess Drusen images 
for filename in os.listdir(directory_drusen):
    if filename.endswith(".jpg"):
        # Open the image
        image = Image.open(os.path.join(directory_drusen, filename))

        # Resize the image with bilinear interpolation
        resized_image = image.resize(new_size, resample=Image.BILINEAR)

        # turn the image into an array
        image_array = np.array(resized_image)

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
        preprocessed_image.save(os.path.join(directory_drusen_processed, new_filename))

# preprocess Exudate images 
for filename in os.listdir(directory_exudate):
    if filename.endswith(".jpg"):
        # Open the image
        image = Image.open(os.path.join(directory_exudate, filename))

        # Resize the image with bilinear interpolation
        resized_image = image.resize(new_size, resample=Image.BILINEAR)

        # turn the image into an array
        image_array = np.array(resized_image)

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
        preprocessed_image.save(os.path.join(directory_exudate_processed, new_filename))

# preprocess Normal images 
for filename in os.listdir(directory_normal):
    if filename.endswith(".jpg"):
        # Open the image
        image = Image.open(os.path.join(directory_normal, filename))

        # Resize the image with bilinear interpolation
        resized_image = image.resize(new_size, resample=Image.BILINEAR)

        # turn the image into an array
        image_array = np.array(resized_image)

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
        preprocessed_image.save(os.path.join(directory_normal_processed, new_filename))
