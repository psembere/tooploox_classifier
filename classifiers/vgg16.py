import os

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from data_set_deserializer import get_data_set
from globals import SERIALIZED_DATA_PATH
from svm_wrappers import FeaturesDataSet, linear_classifier


def get_vgg16_features(cnn_model, img_path, pictures, overwrite=False):
    if os.path.isfile(img_path) and not overwrite:
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


def get_features(overwrite=False):
    cnn_model = VGG16(weights='imagenet', include_top=False)
    data_set = get_data_set()
    feature_data_set = FeaturesDataSet()

    _set_test_features(data_set, feature_data_set, cnn_model, overwrite)
    _set_train_features(data_set, feature_data_set, cnn_model, overwrite)
    return feature_data_set


def _set_train_features(data_set, feature_data_set, cnn_model, overwrite):
    train_features_path = os.path.join(SERIALIZED_DATA_PATH, 'vgg16_train.npy')
    train_features_np = get_vgg16_features(cnn_model, train_features_path, data_set.training_pictures, overwrite)

    feature_data_set.train_features = train_features_np.tolist()
    feature_data_set.train_labels = data_set.training_pictures_labels


def _set_test_features(data_set, feature_data_set, model, overwrite):
    test_features_path = os.path.join(SERIALIZED_DATA_PATH, 'vgg16_test.npy')
    test_features_np = get_vgg16_features(model, test_features_path, data_set.testing_pictures, overwrite)

    feature_data_set.test_features = test_features_np.tolist()
    feature_data_set.test_labels = data_set.testing_pictures_labels


if __name__ == "__main__":
    features = get_features(overwrite=True)

    linear_classifier(features)

    print("successfully ends")
