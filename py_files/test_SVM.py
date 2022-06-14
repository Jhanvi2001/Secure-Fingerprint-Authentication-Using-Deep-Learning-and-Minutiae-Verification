import cv2
from LBP_helper import LocalBinaryPatterns
import joblib

desc = LocalBinaryPatterns(24, 8)
data = []  # Will
imagePath = 'Dataset\\training\\Fake\\2_10.png'
gray = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
data = desc.describe(gray)
model = joblib.load('model\\SVM_rbf.h5')
print(model.predict(data.reshape(1, -1)))
'''
What does it really mean by data.reshape(1, -1)?
Link: https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
'''
