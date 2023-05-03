from PIL import Image
import os

directory = 'Chosen dataset\\Normal'
directory_resized = 'Chosen dataset\\Normal_resized'

# Define the desired size of the resized image
new_size = (224, 224)

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        # Open the image
        image = Image.open(os.path.join(directory, filename))

        # Resize the image with bilinear interpolation
        resized_image = image.resize(new_size, resample=Image.BILINEAR)

        # Save the resized image with a new filename
        new_filename = os.path.splitext(filename)[0] + '_resized.jpg'
        resized_image.save(os.path.join(directory_resized, new_filename))