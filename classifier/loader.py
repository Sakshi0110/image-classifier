import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def load_train_data():
    filenames = os.listdir("data/train")
    np.random.shuffle(filenames)
    for file in filenames:
        image = plt.imread("data/train/" + file)
        label = 0 if "cat" in file else 1
        image = tf.image.resize(image / 255, (64, 64))
        image = np.array([image])
        label = np.array([label])
        yield image, label


def load_test_data():
    filenames = os.listdir("data/test")
    np.random.shuffle(filenames)
    for file in filenames:
        image = plt.imread("data/test/" + file)
        image = tf.image.resize(image / 255, (64, 64))
        yield image


if __name__ == "__main__":
    g = load_test_data()
    x = next(g)
    plt.imshow(x)
    plt.show()


