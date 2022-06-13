import os

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from random import randint
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = ""
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        temp = randint(0, 100)
        if temp % 2 == 0:
            data = "Real"
        else:
            data = "Fake"
        return render_template('index.html', data=data, filenames=f.filename)
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
