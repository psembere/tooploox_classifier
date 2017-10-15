import os

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as keras_image

from data_set_deserializer import get_data_set
from t_sne import visualize_features_sklearn_tsne, visualize_features_tsne
from globals import SERIALIZED_DATA_PATH
from svm_wrappers import FeaturesDataSet, linear_classifier


class Vgg16FeatureDataSet(object):
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)

    def get_features(self, overwrite=False):
        data_set = get_data_set()
        feature_data_set = FeaturesDataSet()

        self._set_test_features(data_set, feature_data_set, overwrite)
        self._set_train_features(data_set, feature_data_set, overwrite)
        return feature_data_set

    def _set_train_features(self, data_set, feature_data_set, overwrite):
        train_features_path = os.path.join(SERIALIZED_DATA_PATH, 'vgg16_train.npy')
        train_features_np = self._get_vgg16_features(train_features_path, data_set.training_pictures, overwrite)

        feature_data_set.train_features = train_features_np.tolist()
        feature_data_set.train_labels = data_set.training_pictures_labels

    def _set_test_features(self, data_set, feature_data_set, overwrite):
        test_features_path = os.path.join(SERIALIZED_DATA_PATH, 'vgg16_test.npy')
        test_features_np = self._get_vgg16_features(test_features_path, data_set.testing_pictures, overwrite)

        feature_data_set.test_features = test_features_np.tolist()
        feature_data_set.test_labels = data_set.testing_pictures_labels

    def _get_vgg16_features(self, features_path, pictures, overwrite):
        if os.path.isfile(features_path) and not overwrite:
            return np.load(features_path)
        else:
            vgg_features = np.array([self._get_cnn_code(idx, image) for idx, image in enumerate(pictures)])
            np.save(features_path, vgg_features)
            return vgg_features

    def _get_cnn_code(self, idx, image):
        if idx % 1000 == 0:
            print(idx)
        img_converted = keras_image.img_to_array(image)
        img_expanded = np.expand_dims(img_converted, axis=0)
        x = preprocess_input(img_expanded)
        return self.model.predict(x)[0][0][0]


if __name__ == "__main__":
    features = Vgg16FeatureDataSet().get_features(overwrite=False)
    #linear_classifier(features)
    #visualize_features_tsne(features)
    visualize_features_sklearn_tsne(features)
    print("successfully ends")
