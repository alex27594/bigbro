import numpy as np
import cv2

from sklearn.cluster import KMeans
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from collections import Counter

SCALE = 5

def change_image_by_mean(image, X_pred, links):
    new_image = np.empty([image.shape[0], image.shape[1], 3])
    already_counted_means = {}
    for i in range(0, X_pred.shape[0]):
        if X_pred[i] not in already_counted_means.keys():
            average_colors = [0, 0, 0]
            average_colors[0] = np.mean([image[links[j][0]][links[j][1]][0] for j in range(0, X_pred.shape[0]) if X_pred[j] == X_pred[i]])
            average_colors[1] = np.mean([image[links[j][0]][links[j][1]][1] for j in range(0, X_pred.shape[0]) if X_pred[j] == X_pred[i]])
            average_colors[2] = np.mean([image[links[j][0]][links[j][1]][2] for j in range(0, X_pred.shape[0]) if X_pred[j] == X_pred[i]])
            new_image[links[i][0], links[i][1]][0] = average_colors[0]
            new_image[links[i][0], links[i][1]][1] = average_colors[1]
            new_image[links[i][0], links[i][1]][2] = average_colors[2]
            already_counted_means[X_pred[i]] = average_colors
        else:
            new_image[links[i][0], links[i][1]][0] = already_counted_means[X_pred[i]][0]
            new_image[links[i][0], links[i][1]][1] = already_counted_means[X_pred[i]][1]
            new_image[links[i][0], links[i][1]][2] = already_counted_means[X_pred[i]][2]
    return new_image


image = imread('/home/alexander/PycharmProjects/car_detection/clean_dirty_image/dirty/9f074e8s-960.jpg')
# image = imread('/home/alexander/PycharmProjects/car_detection/clean_dirty_image/clean/3bf5314s-960.jpg')
image = imresize(image[int(image.shape[0]/2):, :, :], (int(image.shape[0]/SCALE), int(image.shape[1]/SCALE)), interp="nearest")
print(image.shape)
for i in range(0, 3):
    print(image[10][10][i])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print(hsv.shape)
X = np.empty([hsv.shape[0] * hsv.shape[1], 3])
print(X.shape)
links = []
pointer = 0
for i in range(0, hsv.shape[0]):
    for j in range(0, hsv.shape[1]):
        X[pointer][0] = hsv[i][j][0]
        X[pointer][1] = hsv[i][j][1]
        X[pointer][2] = hsv[i][j][2]
        pointer += 1
        links.append((i, j))
clf = KMeans(n_clusters=10, init="k-means++", random_state=241)
X_pred = clf.fit_predict(X)
new_hsv = change_image_by_mean(hsv, X_pred, links).astype(np.uint8)


print(set([i for i in new_hsv[:, :, 0].flatten() if 110 < i and i < 119]))
print(Counter(new_hsv[:, :, 0].flatten().tolist()))

new_image = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
print(new_image.shape)
print(new_image)
imsave("real.jpg", image)
imsave("after.jpg", new_image)
