from flask import Flask, flash, request, redirect
import requests
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from preprocessor import Preprocessor
prep = Preprocessor([128,32],data_augmentation=False)

import os
import json
import io
import base64
from PIL import Image
import numpy as np

@app.route('/setprep',methods=['POST'])
def initprep():
    input_json = request.json
    # print(input_json)
    # rd = json.loads(str(input_json))
    rd = input_json
    args = rd['args']
    kwargs = rd['kwargs']
    prep = Preprocessor(*args,**kwargs)
    return f"Okay, {args} and {kwargs} were sent"

@app.route('/processimage',methods=["POST"])
def processimage():
    if 'file' not in request.files:
        # flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        b = io.BytesIO()
        file.save(b)
        img = Image.open(b).convert('L')
        
        enc_img = base64.b64encode(img)
        
        img = np.array(img)
        processedimg = prep.process_img(img)
        r = requests.post('http://infer:5001/infer',json={'img':enc_img,'shape':processedimg.shape})
        return r.text
    return
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)