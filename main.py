from flask import Flask, redirect, url_for, request, jsonify
from get_image import get_image

app = Flask(__name__)


@app.route('/')
def home():
    return "hello world"

@app.route('/score', methods = ['GET', 'POST'])
def score():
    if request.method == 'GET':
        img_x = request.args.get('x')
        img_y = request.args.get('y')
        image_content = get_image(img_x, img_y)
        image_content = image_content.encode('base64')
        res_model = request.post('http://0.0.0.0:8080/model', model_input = image_content)
        res_module = request.post('http://0.0.0.0:8080/module', module_input = res_model)
        return res_module
	
@app.route('/image', methods = ['GET','POST'])
def image():
	if request.method == 'GET':
        img_x = request.args.get('x')
        img_y = request.args.get('y')
        image_content = get_image(img_x, img_y)
        image_content = image_content.encode('base64')
        res = request.post('http://0.0.0.0:8080/model', model_input = image_content)
	    return res
    else :
        return 'not using get method'

@app.route('/categories', methods = ['GET', 'POST'])
def categories():
    if request.method == 'GET':
        img_x = request.args.get('x')
        img_y = request.args.get('y')
        image_content = get_image(img_x, img_y)
        image_content = image_content.encode('base64')
        res_model = request.post('http://0.0.0.0:8080/model', model_input = image_content)
        res_module = request.post('http://0.0.0.0:8080/categories', module_input = res_model)
        return res_module

if __name__ == "__main__" :
    app.run(host='0.0.0.0', port=8080, debug=True)


