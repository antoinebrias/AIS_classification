#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:20:51 2024

@author: antoine
"""
import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 2: Load and Preprocess Your Data
train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]


# Split your data into training and testing sets
train_data_dir = './output'  # Path to the directory containing all data
all_images = os.listdir(train_data_dir)
#train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=88)




train_generator = train_datagen.flow_from_directory(
    train_data_dir,  # Path to the directory containing training images
    target_size=(128, 128),    # Resize images to 150x150
    batch_size=16,             # Number of images to process in each batch
    class_mode='categorical')

# Determine the number of classes (number of subfolders)
num_classes = len(os.listdir('./output'))


# Get a batch of images and labels from the training generator
images, labels = train_generator.next()

# Display the first few images and their corresponding labels
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title('Label: {}'.format(labels[i]))
    plt.axis('off')
plt.show()



# Step 3: Define Your CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Use num_classes for the output layer
])

# Step 4: Compile Your Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use 'categorical_crossentropy' for multi-class classification
              metrics=['accuracy'])

# Step 5: Train Your Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10
)

# Step 6: Evaluate Your Model on Test Data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    './test',      # Path to the directory containing test images
    target_size=(128, 128),   # Resize images to 150x150
    batch_size=1,            # Number of images to process in each batch
    class_mode='categorical'  # Use 'categorical' mode for multi-class classification
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy:.2f}')



# Generate predictions for the test set
predictions = model.predict(test_generator)

# Get a batch of images and labels from the test generator
images, labels = test_generator.next()

# Select a subset of images from the batch (e.g., first 4 images)
subset_images = images[:4]
subset_labels = labels[:4]
subset_predictions = predictions[:4]

# Get class labels
class_labels = list(test_generator.class_indices.keys())

# Plot the images along with their predicted labels and actual labels
plt.figure(figsize=(10, 10))
for i in range(len(subset_images)):
    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(subset_images[i])
    
    # Get the predicted label and actual label
    predicted_label = np.argmax(subset_predictions[i])
    actual_label = np.argmax(subset_labels[i])
    
    plt.title(f'Predicted: {class_labels[predicted_label]}\nActual: {class_labels[actual_label]}')
    plt.axis('off')

plt.tight_layout()
plt.show()


