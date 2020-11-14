import numpy as np
import cv2
from Network import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm
import statistics
from statistics import mode
from collections import Counter
from statistics import mean

FLOWER_DAISY_DIR = r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\flower_photos\daisy'
FLOWER_SUNFLOWER_DIR = r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\flower_photos\sunflowers'
FLOWER_TULIP_DIR = r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\flower_photos\tulips'
FLOWER_DANDI_DIR = r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\flower_photos\dandelion'
FLOWER_ROSE_DIR = r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\flower_photos\roses'

sorted = os.listdir(FLOWER_DAISY_DIR)
sorted.sort()
len(sorted[:-100])

pixels = 100

X = []
Y = []
X_test = []
Y_test = []


def assign_label(img, flower_type):
    return flower_type


def make_train_test_data(flower_type, DIR):
    sorted = os.listdir(DIR)
    sorted.sort()
    for train_img in tqdm(sorted[-100:]):
        label = assign_label(train_img, flower_type)
        path = os.path.join(DIR, train_img)
        ti = cv2.imread(path, cv2.IMREAD_COLOR)  # IMREAD_GRAYSCALE for grayscale
        ti = cv2.resize(ti, (pixels, pixels))

        X_test.append(np.array(ti))
        Y_test.append(str(label))

    len(sorted[:-100])  # all but last 100 images in file
    for img in tqdm(sorted[:-100]):
        label = assign_label(img, flower_type)
        path = os.path.join(DIR, img)
        i = cv2.imread(path, cv2.IMREAD_COLOR)  # IMREAD_GRAYSCALE for grayscale
        i = cv2.resize(i, (pixels, pixels))

        X.append(np.array(i))
        Y.append(str(label))


make_train_test_data('Daisy', FLOWER_DAISY_DIR)
print(len(X))
make_train_test_data('Rose', FLOWER_ROSE_DIR)
print(len(X))
make_train_test_data('Sunflower', FLOWER_SUNFLOWER_DIR)
print(len(X))
make_train_test_data('Tulip', FLOWER_TULIP_DIR)
print(len(X))
make_train_test_data('Dandelion', FLOWER_DANDI_DIR)
print(len(X))

XX = []
XX_test = []
for image in X:
    XX.append(image.reshape(pixels * pixels * 3))
for image in X_test:
    XX_test.append(image.reshape(pixels * pixels * 3))
XX_np = np.array(XX)
XX_test_np = np.array(XX_test)
Y_test_np = (np.array(Y_test))
Y_np = np.array(Y)

from sklearn.utils import shuffle

X_shuffled, Y_shuffled = shuffle(XX_np, Y_np, random_state=0)

X_shuffled_test, Y_shuffled_test = shuffle(XX_test_np, Y_test_np, random_state=0)

# change to onehot encoded
import pandas as pd

Y_one_hot = pd.get_dummies(Y_shuffled)
Y_test_one_hot = pd.get_dummies(Y_shuffled_test)

Y_train = Y_one_hot.to_numpy()
Y_test = Y_test_one_hot.to_numpy()

model = Network(X_shuffled, Y_train, 700, 3)

testingAccuracies = []
model.feedforward()
model.backprop()
while model.errors[-1] > 0.2:
    model.feedforward()
    model.backprop()


def get_acc(x, y):
    acc = 0
    class_acc = [0, 0, 0, 0, 0]
    for xx, yy in zip(x, y):
        s = model.predict(xx)
        # print(s, np.argmax(yy))
        if s == np.argmax(yy):
            acc += 1
            class_acc[s] += 1
    return acc / len(x) * 100, class_acc


plt.figure(1)
plt.plot(model.errors)
print("Training accuracy : ", get_acc(X_shuffled, Y_train)[0])
a = get_acc(X_shuffled_test, Y_test)
print("Test accuracy : ", a[0])
print("Test accuracy per class : ", a[1])
import pickle

with open('model.pkl', 'wb') as output:
    pickle.dump(model, output)
