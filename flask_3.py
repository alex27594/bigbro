import numpy as np
import json

from flask import Flask
from flask import request
from flask import Response
from scipy.ndimage.filters import gaussian_filter
from keras.models import load_model
from scipy.misc import imresize
from scipy.misc import imread


BRAND_MODEL = load_model("weights-improvement-07-0.78.hdf5")

app = Flask(__name__)

dictionary = {
    0: "porsche",
    1: "renault"
}


@app.route('/', methods=['GET'])
def predict_brand():
    path = request.args.get('path')
    top = int(request.args.get('top'))
    left = int(request.args.get('left'))
    bottom = int(request.args.get('bottom'))
    right = int(request.args.get('right'))
    x = imread(path)
    x = x[top: bottom, left:right]
    x = gaussian_filter(x, 3)
    x = imresize(x, (224, 224))
    x = np.reshape(x, [1, 224, 224, 3])
    x = x / 255
    answer = {"model": dictionary[sorted(list(enumerate(BRAND_MODEL.predict(x)[0].tolist())), key=lambda item: item[1])[-1][0]]}
    return Response(response=json.dumps(answer),
                    status=200,
                    mimetype="application/json")

if __name__ == "__main__":
    app.run(port=5002)
