from model import Model, DecoderType
from imp_model import Modal
from dataloader_iam import Batch

from flask import Flask, request
from werkzeug.utils import secure_filename
app = Flask(__name__)

import os
import json
import io
from PIL import Image
from base64 import decodestring
import numpy as np
from main import FilePaths


model = Modal(list(open(FilePaths.fn_char_list).read()), DecoderType.BestPath, must_restore=True)
@app.route('/infer',methods=["POST"])
def infer():
    input_json = request.json

    image = Image.fromstring('RGB',(input_json['shape']),decodestring(input_json['img']))
    print(image.shape)
    batch = Batch([image], None, 1)
    recognized, probability = model.infer_batch(batch, True)
   
    return {"Recognized":str(recognized[0]), "Probability":float(probability[0])}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)