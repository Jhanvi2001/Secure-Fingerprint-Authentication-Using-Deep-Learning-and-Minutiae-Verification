# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:38:21 2022

@author: sjhan
@edited: settler
"""
'''
Incase you get an error saying data has not been read then run it in the cmd after activating the virtual environment
'''
from tqdm import tqdm
from LBP_helper import LocalBinaryPatterns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from imutils import paths
import cv2
import os

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []  # Will
labels = []
home_path = os.getcwd()
import glob
# loop over the training images
# imagePaths = list(paths.list_images('Dataset\\training')) + list(paths.list_images('Dataset\\testing'))
imagePaths = glob.iglob('Dataset/' + '**/*.png', recursive=True)
# print(list(imagePaths))
# print(list(paths.list_images('testing')))
print("Done with storing the list of images")

print("Started image processing")
for imagePath in tqdm(imagePaths):
    # load the image, convert it to grayscale, and describe it
    gray = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    hist = desc.describe(gray)

    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)
print("Done with image processing")

'''
Split the data into a training and testing set
Train: 80%
Test: 20%
'''
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

'''
Performing GridSearchCV with 5 folds to get the best value of parameters
'''
grid_search = GridSearchCV(SVC(kernel='rbf'),
                           param_grid={'C': [10, 100, 1000, 10000, 100000], 'gamma': [100, 10, 1, 0.1, 0.01, .0001]},
                           cv=5, verbose=True)
grid_search.fit(x_train, y_train)
params = grid_search.best_params_
print("Best parameter: ", params)

model = SVC(kernel='rbf', C=params['C'], gamma=params['gamma'], probability=True,verbose=True)
model.fit(x_train, y_train)

#save the model with joblibpython 
import joblib
filename = 'model/SVM_rbf.h5'
joblib.dump(model, filename)

model.score(x_test, y_test)
print(model.predict(x_test[1].reshape(1, -1)))
print(model.predict_proba(x_test[1].reshape(1, -1)))
print("Test accuracy: ", model.score(x_test, y_test))
print("train accuracy: ", model.score(x_train, y_train))

# check image
# import numpy as np
# from keras.preprocessing import image

# test_image = cv2.imread("Fake_1.png")
# gray1 = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# hist1 = desc.describe(gray1)
#
# result = model.predict(hist1.reshape(1, -1))
# if result[0][0] == 1:
#     prediction = 'Live'
# else:
#     prediction = 'Fake'
# print(result)
