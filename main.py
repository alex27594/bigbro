import os
import json
import numpy as np

from flask import Flask
from flask import request
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from yolo import yolo_eval, yolo_head


MODEL_PATH = os.path.expanduser("model_data/yolo.h5")
ANCHORS_PATH = os.path.expanduser("model_data/yolo_anchors.txt")
CLASSES_PATH = os.path.expanduser("model_data/coco_classes.txt")
SCORE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
SEARCHED_CLASS_NAMES = {"bicycle", "car", "motorbike",
                        "bus", "train", "truck",
                        "suitcase", "cell phone"}

sess = K.get_session()

with open(CLASSES_PATH) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

with open(ANCHORS_PATH) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

yolo_model = load_model(MODEL_PATH)

model_image_size = yolo_model.layers[0].input_shape[1:3]
is_fixed_size = model_image_size != (None, None)

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(
    yolo_outputs,
    input_image_shape,
    score_threshold=SCORE_THRESHOLD,
    iou_threshold=IOU_THRESHOLD)

app = Flask(__name__)


@app.route('/', methods=['GET'])
def predict():
    path = "images/" + request.args.get('path')
    res = []
    image = Image.open(path)
    if is_fixed_size:
        resized_image = image.resize(
            tuple(reversed(model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        resized_image = image.resize(new_image_size, Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        #print(image_data.shape)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })
    #print('Found {} boxes for {}'.format(len(out_boxes), path))

    for i in range(len(out_classes)):
        if class_names[out_classes[i]] in SEARCHED_CLASS_NAMES:
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            res.append({"top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right)})

    return json.dumps(res)

if __name__ == "__main__":
    app.run()