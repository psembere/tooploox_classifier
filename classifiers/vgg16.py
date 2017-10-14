from keras.preprocessing import image

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

import numpy as np

from utils.data_loader import get_data_set
from utils.globals import PROJECT_PATH
import os


def get_vgg16_features(reload=False):
    img_path = os.path.join(PROJECT_PATH, 'data', 'vgg16.npy')
    a = os.path.isfile(img_path)
    if os.path.isfile(img_path) and not reload:
        features = np.load(img_path)
    else:
        features = generate_vgg16_features(img_path)
    return features


def generate_vgg16_features(img_path):
    features = []
    idx = 0
    model = VGG16(weights='imagenet', include_top=False)
    data_set = get_data_set()
    for img in data_set.training_pictures:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model.predict(x)[0][0][0])
        print(idx)
        idx += 1
    np.save(img_path, features)
    return features


if __name__ == "__main__":
    features = get_vgg16_features()

    print(features)
