import base64
import os
from flask_cors import CORS
from flask import Flask, request
import cv2
import stitcher

app = Flask(__name__)
stitcher = stitcher.Stitcher()
CORS(app)

@app.route('/init', methods=['GET'])
def hello_world():
    print ('---init received')
    return 'Service is up and running.<br>It simply stitches two images into one.<br>Make sure two images has an overlap!'

@app.route('/api/stitch', methods=['POST'])
def stitich_image():
    image_a = request.files['left'].save(os.path.join(app.root_path, 'images', '1.JPG'))
    image_b = request.files['right'].save(os.path.join(app.root_path, 'images', '2.JPG'))

    image_a = cv2.imread('images/1.JPG')
    image_b = cv2.imread('images/2.JPG')
    result = stitcher.stitch([image_a, image_b])
    cv2.imwrite('images/result.JPG', result)
    print('---stitch complete')
    with open('images/result.JPG', 'rb') as result_img:
        result = result_img.read()
        result = base64.b64encode(result)
    return result

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
