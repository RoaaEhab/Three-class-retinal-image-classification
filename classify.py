from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.svm import SVC

# Load a pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Define the layer of interest to extract features
layer_name = 'block5_conv3'
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

# Preprocess the images using the VGG16 preprocessing function
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Generate features and labels from the training images
train_generator = datagen.flow_from_directory('Chosen dataset\Processed folders', target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)
features = model.predict(train_generator, verbose=1)

# Save the features and labels to numpy arrays
np.save('./features.npy', features)
np.save('./labels.npy', train_generator.classes)





# # Load the pre-trained VGG16 model
# base_model = VGG16(weights='imagenet', include_top=False)

# # Define the layer of interest to extract features
# layer_name = 'block5_conv3'
# model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

# # Load the preprocessed features and labels from the images
# features = np.load('./features.npy')
# labels = np.load('./labels.npy')

# # Flatten the features into one-dimensional vectors
# features = features.reshape(features.shape[0], -1)

# # Train an SVM classifier on the flattened features
# clf = SVC(kernel='linear', C=1.0, probability=True)
# clf.fit(features, labels)

# # Load and preprocess an image to be classified
# image_path = 'Chosen dataset\Drusen_preprocessed\AWM_2007-07-02_OD_88341_resized_processed.jpg'
# image = load_img(image_path, target_size=(224, 224))
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)
# image = preprocess_input(image)

# # Extract features from the image using the pre-trained VGG16 model
# features = model.predict(image)

# # Classify the image using the SVM classifier
# predicted_label = clf.predict(features)[0]

# # Map the predicted label to a class name
# class_names = ['Normal', 'Drusen', 'Exudate', ...]  # replace with your class names
# predicted_class_name = class_names[predicted_label]

# # Output the classification result
# print(f"The image is classified as: {predicted_class_name}")
