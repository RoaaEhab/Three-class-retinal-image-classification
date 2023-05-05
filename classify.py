from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import normalize


#Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

#Define the layer of interest to extract features
layer_name = 'block5_conv3'
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

# Load the preprocessed features and labels from the images
features = np.load('./features.npy')
labels = np.load('./labels.npy')

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Flatten the features into one-dimensional vectors
features_train = features_train.reshape(features_train.shape[0], -1)
features_test = features_test.reshape(features_test.shape[0], -1)

# Train an SVM classifier on the flattened features
clf = SVC(kernel='linear', C=1.0, probability=True)
clf.fit(features_train, labels_train)


class_names = ['Drusen', 'Exudate', 'Normal', ...] 
predicted_class_name=[]
# Classify the image using the SVM classifier
for i in range(0, len(features_test)):
    predicted_label = clf.predict(features_test)[i]
    predicted_class_name.append(predicted_label)

# Output the classification result   
print(predicted_class_name)   
print(labels_test)

cm = confusion_matrix(labels_test, predicted_class_name)
print(cm)

# Plot the confusion matrix
# plot_confusion_matrix(clf, features_test, labels_test, cmap=plt.cm.Blues)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()

# Calculate the sensitivity
sensitivity = recall_score(labels_test, predicted_class_name, average='macro')

# Extract the true negatives (TN) and false positives (FP) for each class
tn_0 = np.sum(cm[1:, 1:])  # TN for class 0
fp_0 = np.sum(cm[1:, 0])   # FP for class 0
tn_1 = np.sum(np.vstack((cm[0, 0], cm[2, 2])))  # TN for class 1
fp_1 = np.sum(np.hstack((cm[0, 1:], cm[2, :2])))  # FP for class 1
tn_2 = np.sum(cm[:2, :2])  # TN for class 2
fp_2 = np.sum(np.hstack((cm[:2, 2], cm[2, 0:2])))  # FP for class 2

# Calculate the specificity for each class
spec_0 = tn_0 / (tn_0 + fp_0)
spec_1 = tn_1 / (tn_1 + fp_1)
spec_2 = tn_2 / (tn_2 + fp_2)

# Calculate the accuracy score
accuracy = accuracy_score(labels_test, predicted_class_name)

# Calculate the F-score
f_score = f1_score(labels_test, predicted_class_name, average='weighted')

# Calculate the AUC score
y_test = np.reshape(labels_test, (-1, 1))
y_pred = np.reshape(predicted_class_name, (-1, 1))
y_pred = normalize(y_pred, axis=1, norm='l1')
y_test = normalize(y_test, axis=1, norm='l1')

auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr')

print("Sensitivity: {:.2f}".format(sensitivity))

print("Specificity for class Drusen: {:.2f}".format(spec_0))
print("Specificity for class Exudate: {:.2f}".format(spec_1))
print("Specificity for class Normal: {:.2f}".format(spec_2))

print("Accuracy: {:.2f}".format(accuracy))

print("F-score: {:.2f}".format(f_score))

print("AUC: {:.2f}".format(auc_score))

# show the confusion matrix plot
plt.show()
