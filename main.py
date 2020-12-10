from flask import Flask, redirect, url_for, request
from get_image import get_image

app = Flask(__name__)

def classify(image):
    #result = call module
    return result

@app.route('/')
def home():
    return "fuck!"

@app.route('/score')
def score():
    img_x = request.args.get('x')
    img_y = request.args.get('y')
    image_content = get_image(img_x, img_y)
    output = classify(image_content)
    #result = score_module(output) / json file
    return result

@app.route('/image')
def image():
    img_x = request.args.get('x')
    img_y = request.args.get('y')
    image_content = get_image(img_x, img_y)
    output = classify(image_content)
    return redirect(url_for('show_img', segmented_image = output))

@app.route('/image/PNG')
def show_img(segmented_image):
    
    return 0
@app.route('/categories')
def categories():
    return "suck!"

if __name__ == "__main__" :
    app.run(host='0.0.0.0', port=8080, debug=True)

#/