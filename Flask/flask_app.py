
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, render_template , request , send_file
import os
from yolo import process_image


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/myname')
def myname():
    return "Your got it"

@app.route('/upload', methods=['POST'])
def upload_file():

    file = request.files['file']

    if 'file' not in request.files :
        return render_template('mistake.html')

    if file.filename == '':
        return render_template('mistake.html')

    if file:
        filename = os.path.join(os.path.dirname(__file__), 'uploads', file.filename)
        file.save(filename)

        #Call image processing function
        processed_filename = process_image(filename)

        return send_file(processed_filename, mimetype='image/jpeg')

