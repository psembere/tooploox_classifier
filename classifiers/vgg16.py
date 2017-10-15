import os

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from data_set_deserializer import get_data_set
from globals import SERIALIZED_DATA_PATH
from svm_wrappers import FeaturesDataSet, linear_classifier


def get_vgg16_features(cnn_model, img_path, pictures, reload=False):
    if os.path.isfile(img_path) and not reload:
        return np.load(img_path)
    else:
        return generate_vgg16_features(cnn_model, img_path, pictures)


def generate_vgg16_features(model, img_path, pictures):
    features = []
    idx = 0
    for img in pictures:
        x = _prepare_picture(img)
        features.append(model.predict(x)[0][0][0])
        if idx % 1000 == 0:
            print(idx)
        idx += 1
    np.save(img_path, features)
    return np.array(features)


def _prepare_picture(img):
    img_converted = image.img_to_array(img)
    img_expanded = np.expand_dims(img_converted, axis=0)
    return preprocess_input(img_expanded)


def get_features(model, reload=False):
    data_set = get_data_set()
    pictures_train = data_set.training_pictures
    pictures_test = data_set.testing_pictures

    img_path_train = os.path.join(SERIALIZED_DATA_PATH, 'vgg16_train.npy')
    img_path_test = os.path.join(SERIALIZED_DATA_PATH, 'vgg16_test.npy')

    features_test = get_vgg16_features(model, img_path_test, pictures_test, reload).tolist()
    features_train = get_vgg16_features(model, img_path_train, pictures_train, reload).tolist()

    features = FeaturesDataSet()
    features.train_features = features_train
    features.test_features = features_test
    features.train_labels = data_set.training_pictures_labels
    features.test_labels = data_set.testing_pictures_labels

    return features


if __name__ == "__main__":
    model = VGG16(weights='imagenet', include_top=False)

    features = get_features(model, reload=False)

    linear_classifier(features)

    print("successfully ends")
