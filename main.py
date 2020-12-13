from flask import Flask, redirect, url_for, request, jsonify
from get_image import get_image

app = Flask(__name__)

import os
import torch

import numpy as np
from PIL import Image

from unet import UnetPipeline
from logic import logic

device = torch.device('cuda')
unet = UnetPipeline()
unet.load('./model_bri_0112_copy_2_wei_3/', device=device)

@app.route('/')
def home():
    return "hello world"

@app.route('/score', methods = ['GET', 'POST'])
def score():
    if request.method == 'GET':
        img_x = request.args.get('x')
        img_y = request.args.get('y')
        image_content = get_image(img_x, img_y)
        file=open("img.png","wb")
        file.write(image_content)
        file.close()
        input_img = np.array(Image.open('img.png').convert('RGB'))[:,:,:3]
        res = unet.predict(input_img, batch_size = 8)
        score = logic(res)
        print(score)
        return jsonify(score)
	
@app.route('/image', methods = ['GET','POST'])
def image():
    if request.method == 'GET':
        img_x = request.args.get('x')
        img_y = request.args.get('y')
        image_content = get_image(img_x, img_y)
        file=open("img.png","wb")
        file.write(image_content)
        file.close()
        input_img = np.array(Image.open('img.png').convert('RGB'))[:,:,:3]
        res = unet.predict(input_img, batch_size = 8)
        res = res.tolist()
        return jsonify(res)#return as 2d list

    else :
         return 'error'

# @app.route('/categories', methods = ['GET', 'POST'])
# def categories():
#     if request.method == 'GET':
#         img_x = request.args.get('x')
#         img_y = request.args.get('y')
#         image_content = get_image(img_x, img_y)
#         image_content = image_content.encode('base64')
#         res_model = request.post('http://0.0.0.0:8080/model', model_input = image_content)
#         res_module = request.post('http://0.0.0.0:8080/categories', module_input = res_model)
#         return res_module

if __name__ == "__main__" :
    app.run(host='0.0.0.0', port=8080, debug=True)


