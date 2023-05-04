import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np

# Load a pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Define the layer of interest to extract features
layer_name = 'block5_conv3'
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

# Load and preprocess an image
image_path = 'Chosen dataset\Drusen_preprocessed\AWM_2007-07-02_OD_88341_resized_processed.jpg'
image = load_img(image_path)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# image = preprocess_input(image)

# Extract features from the image using the pre-trained CNN
features = model.predict(image)

# The features variable now contains the extracted features from the image
print(features.shape)
# normalize the feature vector