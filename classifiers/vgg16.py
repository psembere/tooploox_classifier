import os

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from classifiers.svm_wrappers import FeaturesLabelsDataSet, linear_classifier
from utils.data_loader import get_data_set
from utils.globals import PROJECT_PATH


def get_vgg16_features(model, img_path, pictures, reload=False):
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
        if idx % 10 == 0:
            print(idx)
        idx += 1
    np.save(img_path, features)
    return features


def get_features(model, reload=False):
    data_set = get_data_set()
    pictures_train = data_set.training_pictures
    pictures_test = data_set.testing_pictures

    img_path_train = os.path.join(PROJECT_PATH, 'data', 'vgg16_train.npy')
    img_path_test = os.path.join(PROJECT_PATH, 'data', 'vgg16_test.npy')

    features_test = get_vgg16_features(model, img_path_test, pictures_test, reload)
    features_train = get_vgg16_features(model, img_path_train, pictures_train, reload)

    features_labels_data_set = FeaturesLabelsDataSet()
    features_labels_data_set.train_features = features_train.tolist()
    features_labels_data_set.test_features = features_test.tolist()
    features_labels_data_set.train_labels = data_set.training_pictures_labels
    features_labels_data_set.test_labels = data_set.testing_pictures_labels

    return features_labels_data_set


if __name__ == "__main__":
    model = VGG16(weights='imagenet', include_top=False)

    features = get_features(model, reload=True)

    linear_classifier(features)

    print "successfully ends"
