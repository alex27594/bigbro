import os
import numpy as np
from keras.models import load_model
from scipy.misc import imread, imresize
from scipy.ndimage.filters import gaussian_filter

model = load_model("saved_steps/weights-improvement-07-0.78.hdf5")
directory = "/home/alexander/PycharmProjects/car_detection/brand_image/validation/"

for path in os.listdir(directory):
    x = imread(directory + path)
    x = gaussian_filter(x, 3)
    x = imresize(x, (224, 224))
    x = np.reshape(x, [1, 224, 224, 3])
    x = x / 255
    print(path)
    print(sorted(list(enumerate(model.predict(x)[0].tolist())), key=lambda item: item[1])[-1][0])
    print(model.predict(x))
