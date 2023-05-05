from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.svm import SVC

 #Load a pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

 # Define the layer of interest to extract features
layer_name = 'block5_conv3'
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

 # Preprocess the images using the VGG16 preprocessing function
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

 # Generate features and labels from the training images
train_generator = datagen.flow_from_directory('Processed folders', batch_size=32, class_mode='categorical', shuffle=False)
features = model.predict(train_generator, verbose=1)

 # Save the features and labels to numpy arrays
np.save('./features.npy', features)
np.save('./labels.npy', train_generator.classes)