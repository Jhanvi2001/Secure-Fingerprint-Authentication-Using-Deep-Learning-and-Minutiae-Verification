# # -*- coding: utf-8 -*-
# """
# Created on Thu Jun  9 10:45:46 2022
#
# @author: sjhan
# """
#
# import cv2
# import matplotlib.pyplot as plt
# from skimage.filters import threshold_otsu
# import numpy as np
# from glob import glob
# from scipy import misc
# from matplotlib.patches import Circle, Ellipse
# from matplotlib.patches import Rectangle
# import os
# from PIL import Image
#
# import keras
# from matplotlib import pyplot as plt
# import numpy as np
# import gzip
# from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
# from keras.models import Model
# from keras.optimizers import RMSprop
# from tensorflow.keras.layers import BatchNormalization
#
# data = glob('Real_imgs\*')
# len(data)
#
# import imageio.v2 as imageio
#
# images = []
#
#
# def read_images(data):
#     for i in range(len(data)):
#         img = imageio.imread(data[i])
#         img = np.array(Image.fromarray(img).resize((224, 224)))
#         images.append(img)
#     return images
#
#
# images = read_images(data)
# images_arr = np.asarray(images)
# images_arr = images_arr.astype('float32')
# images_arr.shape
# images_arr = images_arr[..., 0]
#
# # Display the first image in training data
# for i in range(2):
#     plt.figure(figsize=[5, 5])
#     curr_img = np.reshape(images_arr[i], (224, 224))
#     plt.imshow(curr_img, cmap='gray')
#     plt.show()
#
# images_arr = images_arr.reshape(-1, 224, 224, 1)
# np.max(images_arr)
# images_arr = images_arr / np.max(images_arr)
# np.max(images_arr), np.min(images_arr)
#
# from sklearn.model_selection import train_test_split
#
# train_X, valid_X, train_ground, valid_ground = train_test_split(images_arr,
#                                                                 images_arr,
#                                                                 test_size=0.2,
#                                                                 random_state=13)
#
# batch_size = 5
# epochs = 30
# inChannel = 1
# x, y = 224, 224
# input_img = Input(shape=(x, y, inChannel))
#
#
# def autoencoder(input_img):
#     # encoder
#     # input = 28 x 28 x 1 (wide and thin)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)
#
#     # decoder
#     conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 128
#     up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128
#     conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 64
#     up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64
#     decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
#     return decoded
#
#
# autoencoder = Model(input_img, autoencoder(input_img))
# autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop(), metrics=['accuracy'])
#
# autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=1,
#                                     validation_data=(valid_X, valid_ground))
#
# loss = autoencoder_train.history['loss']
# val_loss = autoencoder_train.history['val_loss']
# epochs = range(20)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()
#
# pred = autoencoder.predict(valid_X)
# plt.figure(figsize=(20, 20))
# print("Test Images")
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(valid_ground[i, ..., 0], cmap='gray')
# plt.show()
# plt.figure(figsize=(20, 20))
# print("Reconstruction of Test Images")
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(pred[i, ..., 0], cmap='gray')
# plt.show()
#
####################### new img ########################
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf

# Load saved model
autoencoder = keras.models.load_model('model/Reconstructing.h5')
test_image = tf.keras.utils.load_img('web/static/uploads/4_3.png', target_size=(224, 224))
images_arr1 = np.asarray(test_image)
images_arr1 = images_arr1.astype('float32')
# images_arr1.shape
images_arr1 = images_arr1[..., 0]

images_arr1 = images_arr1.reshape(-1, 224, 224, 1)
images_arr1 = images_arr1 / np.max(images_arr1)
images_arr1 = images_arr1.reshape(-1, 224, 224, 1)
pred1 = autoencoder.predict(images_arr1)
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(1):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images_arr1[i, ..., 0], cmap='gray')
plt.show()
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(1):
    plt.subplot(1, 5, i + 1)
    plt.imshow(pred1[i, ..., 0], cmap='gray')
plt.savefig('web/static/uploads/4_3_reconstructed.png',bbox_inches='tight')
plt.show()
plt.close()

from math import log10, sqrt
import numpy as np


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


# original = cv2.imread("1.png")
# compressed = cv2.imread("2.png", 1)
# value = PSNR(original, compressed)
# print(f"PSNR value is {value} dB")
