from flask import Flask, request, jsonify
from flask_cors import CORS

import sys
import torch
import json
import numpy as np
import PIL.Image as Image

from unet import UnetPipeline
from unet.util import cat_to_rgb

from comflogic import analyze

from vworld import VWorld
from util import convert_base64

app = Flask(__name__)
CORS(app)

# load unet model
unet = UnetPipeline()
unet.load('./model', device=torch.device('cuda'))

# load api key and initialize vworld api
with open('./secrets.json') as secrets_file:
    secrets = json.load(secrets_file)
vworld = VWorld(secrets['vworldApiKey'])

cache = dict()

@app.route('/analyze', methods = ['GET'])
def analysis():
    x = request.args.get('x')
    y = request.args.get('y')
    key_str = f'{x},{y}'

    if key_str not in cache:
        # get original image from vworld
        original_img = vworld.get_image(x, y)
        original_arr = np.array(original_img.convert('RGB'))
        
        # get prediction from segmentation model
        predict_arr = unet.predict(original_arr, batch_size = 8)
        predict_img = Image.fromarray(cat_to_rgb(predict_arr).astype('uint8'))

        # get scores from score module
        scores = analyze(predict_arr)

        # convert images to base64
        original_img_base64 = convert_base64(original_img)
        predict_img_base64 = convert_base64(predict_img)

        response = {
            'original_image': original_img_base64,
            'predict_image': predict_img_base64,
            'scores': scores,
        }

        cache[key_str] = response

    return cache[key_str]

if __name__ == "__main__" :
    
    app.run(host='0.0.0.0', port=sys.argv[1], debug=True)
