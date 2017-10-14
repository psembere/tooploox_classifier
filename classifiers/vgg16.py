from keras.preprocessing import image

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

import numpy as np

from utils.data_loader import get_data_set
from utils.globals import PROJECT_PATH
import os


def get_vgg16_features(model, img_path, pictures, reload=False, ):
    if os.path.isfile(img_path) and not reload:
        features = np.load(img_path)
    else:
        features = generate_vgg16_features(model, img_path, pictures)
    return features


def generate_vgg16_features(model, img_path, pictures):
    features = []
    idx = 0
    for img in pictures:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model.predict(x)[0][0][0])
        print(idx)
        idx += 1
    np.save(img_path, features)
    return features


if __name__ == "__main__":
    model = VGG16(weights='imagenet', include_top=False)

    data_set = get_data_set()
    pictures_train = data_set.training_pictures
    pictures_test = data_set.testing_pictures

    img_path_train = os.path.join(PROJECT_PATH, 'data', 'vgg16_train.npy')
    img_path_test = os.path.join(PROJECT_PATH, 'data', 'vgg16_test.npy')


    features_test = get_vgg16_features(model, img_path_test, pictures_test)
    features_train = get_vgg16_features(model, img_path_train, pictures_train)

    # print(features)
