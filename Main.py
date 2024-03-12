import cv2
import os
import numpy as np
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# Define image directory paths
image_directory = "Dataset/"
INPUT_SIZE = 64
# Check if "Datasets" directory exists
if not os.path.exists(image_directory):
    os.makedirs(image_directory)  # Create the directory if it doesn't exist

# Define subfolder paths (assuming they are within "Datasets")
no_tumor_path = os.path.join(image_directory, "no/")
yes_tumor_path = os.path.join(image_directory, "yes/")

# Initialize empty lists for data and labels
Dataset = []
label = []

# Process images without tumors
for image_name in os.listdir(no_tumor_path):
    if image_name.endswith(".jpg"):  # Check for all jpg extensions
        image = cv2.imread(os.path.join(no_tumor_path, image_name))
        image = Image.fromarray(image, "RGB")
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        Dataset.append(np.array(image))
        label.append(0)

# Process images with tumors
for image_name in os.listdir(yes_tumor_path):
    if image_name.endswith(".jpg"):  # Check for all jpg extensions
        image = cv2.imread(os.path.join(yes_tumor_path, image_name))
        image = Image.fromarray(image, "RGB")
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        Dataset.append(np.array(image))
        label.append(1)

# Convert lists to NumPy arrays
Dataset = np.array(Dataset)
label = np.array(label)

# Print shapes of data and labels
print("Dataset shape:", Dataset.shape)
print("Label shape:", label.shape)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(Dataset, label, test_size=0.2, random_state=0)

# Normalize the pixel values (usually between 0 and 1)
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Now you have pre-processed training and testing data (x_train, y_train, x_test, y_test) ready for training your machine learning model.


#############Model Building

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Binary CrossEntropy = 1, Sigmoid
#categorical_crossEntropy =2, softmax

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
batch_size = 16,
verbose = 1, epochs = 10,
validation_data = (x_test, y_test),
shuffle = False)

model.save('BrainTumor10Epochs.h5')