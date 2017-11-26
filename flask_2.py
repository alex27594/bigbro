import numpy as np
import cv2
import json

from flask import Flask
from flask import request
from flask import Response

app = Flask(__name__)


@app.route('/', methods=['GET'])
def predict():
    path1 = request.args.get('path1')
    path2 = request.args.get('path2')

    top = (int)(request.args.get('top'))
    left = (int)(request.args.get('left'))
    bottom = (int)(request.args.get('bottom'))
    right = (int)(request.args.get('right'))

    return check_for_stealing(left, top, right, bottom, path1, path2)


def dist(x1, y1, x2, y2):
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))


def is_trying_to_steal(global_shape, p1, st, p0):
    s1, s2, _ = global_shape
    global_dist = dist(s1, 0, s2, 0)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # print('global_dist', global_dist)

    number_of_points_with_long_dist = 0
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        if (dist(a, b, c, d) / global_dist > 0.1):
            number_of_points_with_long_dist += 1

    # print('number_of_points_with_long_dist', number_of_points_with_long_dist)
    # print(len(p0[st == 1]))

    if (len(p0[st == 1]) == 0):
        return True

    return ((float)(number_of_points_with_long_dist) / len(p0[st == 1]) > 0.3)


def cut_rectangle(rect_left, rect_top, rect_right, rect_bottom, img):
    s1, s2, _ = img.shape
    cuted_img = img[rect_top:rect_bottom, rect_left:rect_right, :]

    return cuted_img


def check_for_stealing(rect_left, rect_top, rect_right, rect_bottom, first_img_path, second_img_path):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=3000,
                          qualityLevel=0.3,
                          minDistance=2,
                          blockSize=5)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(21, 21),
                     maxLevel=10,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    first_img = cv2.imread(first_img_path)

    old_frame = cut_rectangle(rect_left, rect_top, rect_right, rect_bottom, first_img)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if (p0 is None):
        status = {"status": "SOS"}

        return Response(response=json.dumps(status),
                        status=200,
                        mimetype="application/json")

    second_img = cv2.imread(second_img_path)
    frame = cut_rectangle(rect_left, rect_top, rect_right, rect_bottom, second_img)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if (is_trying_to_steal(frame.shape, p1, st, p0)):
        status = {"status": "SOS"}
    else:

        frame_diff = cv2.absdiff(old_gray, frame_gray)
        frame_diff[frame_diff < 50] = 0
        s1, s2 = frame_diff.shape
        s = np.sum(frame_diff != 0)
        res = (float)(s) / (s1 * s2)

        if (res > 0.2):
            status = {"status": "SOS"}
        else:
            status = {"status": "OK"}

    return Response(response=json.dumps(status),
                    status=200,
                    mimetype="application/json")


if __name__ == "__main__":
    app.run(port=5001)
