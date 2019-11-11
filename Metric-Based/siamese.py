# omniglot data: https://github.com/brendenlake/omniglot/tree/master/python

import os
from PIL import Image
import numpy as np

DATA_DIR = "../data/omniglot/images_background"
im_height, im_width = 28, 28
n_examples = 20

def get_train_data():
    # load train dataset
    root_old = ""
    train_data = []
    for root, dirs, files in os.walk(DATA_DIR, topdown=True):
        class_data = np.empty([1, n_examples, im_height, im_width], dtype=np.float32)
        i = -1
        for i, file in enumerate(files):
            if i >= n_examples:
                break
            img_path = os.path.join(root, file)
            img = 1. - np.array(
                Image.open(img_path).resize((im_width, im_height)),
                np.float32, copy=False)
            class_data[0, i] = img
        train_data.append(class_data)
    return np.concatenate(train_data, axis=0)

get_train_data()