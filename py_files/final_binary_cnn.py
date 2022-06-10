# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 18:32:32 2022

@author: sjhan
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

train_path = 'Dataset/training/'
test_path = 'Dataset/testing/'

# data generator

train_data = ImageDataGenerator(rescale=1. / 255)
train_set = train_data.flow_from_directory(directory=train_path, target_size=(128, 128), batch_size=32,
                                           color_mode="rgb", class_mode='binary')

test_data = ImageDataGenerator(rescale=1. / 255)
test_set = train_data.flow_from_directory(directory=test_path, target_size=(128, 128), batch_size=32, color_mode="rgb",
                                          class_mode='binary')



model = Sequential()

# first layer
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# second layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flattern layer
model.add(Flatten())

# Dense layer
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_history = model.fit(train_set, validation_data=test_set, epochs=10)


plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, 10), model_history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 10), model_history.history["val_accuracy"], label="val_accuracy")

plt.title("Training and validation Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

# # check image
#
# test_image = image.load_img('Fake.png', target_size=(128, 128))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = model.predict(test_image)
# # train_set.class_indices
# if result[0][0] == 1:
#     prediction = 'Live'
# else:
#     prediction = 'Fake'
# print(prediction)
