import numpy as np
import cv2

from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def predict():
    path = request.args.get('path')
    pass

def dist(x1, y1, x2, y2):
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))


def check_alarm(global_shape, p1, st, p0):
    s1, s2, _ = global_shape
    global_dist = dist(s1, 0, s2, 0)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    number_of_points_with_long_dist = 0
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        print(dist(a, b, c, d))

        if (dist(a, b, c, d) / global_dist > 0.1):
            number_of_points_with_long_dist += 1

    return ((float)(number_of_points_with_long_dist) / len(p0[st == 1]) > 0.1)


def cut_rectangle(rect_left, rect_top, rect_right, rect_bottom, img):
    s1, s2, _ = img.shape
    cuted_img = img[rect_top:rect_bottom, rect_left:rect_right, :]

    return cuted_img


def check_for_stealing(rect_left, rect_top, rect_right, rect_bottom, first_img_path, second_img_path):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.3,
                          minDistance=3,
                          blockSize=3)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(21, 21),
                     maxLevel=10,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    first_img = cv2.imread(first_img_path)

    old_frame = cut_rectangle(rect_left, rect_top, rect_right, rect_bottom, first_img)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    second_img = cv2.imread(second_img_path)
    frame = cut_rectangle(rect_left, rect_top, rect_right, rect_bottom, second_img)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    return check_alarm(frame.shape, p1, st, p0)