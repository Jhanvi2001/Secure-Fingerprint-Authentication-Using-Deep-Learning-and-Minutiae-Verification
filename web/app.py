import os
import cv2

# from py_files.LBP_helper import LocalBinaryPatterns
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from random import randint
from werkzeug.utils import secure_filename


# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:38:08 2022

@author: sjhan
"""

# import the necessary packages
from skimage import feature
import numpy as np


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns

        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist


desc = LocalBinaryPatterns(24, 8)
data = []  # Will


UPLOAD_FOLDER = 'web\\static\\uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        imagePath = os.path.join(UPLOAD_FOLDER, f.filename)
        gray = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        data = desc.describe(gray)
        model = joblib.load('model\\SVM_rbf.h5')
        # print(model.predict(data.reshape(1, -1)))


        # temp = randint(0, 100)
        # if temp % 2 == 0:
        #     data = "Real"
        # else:
        #     data = "Fake"
        return render_template('index.html', filenames=model.predict(data.reshape(1, -1)))
    else:
        return render_template('index.html')


@app.route('/matching', methods=['POST', 'GET'])
def matching():
    if request.method == 'POST':
        data = jsonify(request.form)
        # print(data)
        # return render_template(url_for('classifier'), data=data)
        return render_template('index.html', data=data)
    else:
        return render_template('matching.html')


@app.route('/reconstruction')
def reconstruction():
    return render_template('reconstruction.html')


if __name__ == '__main__':
    app.run(debug=True, port=3000)
