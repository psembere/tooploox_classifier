import os

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as keras_image

from data_set_deserializer import get_data_set
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


def visualize_features(features):
    import numpy as np
    from matplotlib import pyplot as plt
    from tsne import bh_sne

    # load up data
    x_data = features.train_features
    y_data = features.train_labels

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(x_data).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    # For speed of computation, only run on a subset
    n = 10000
    x_data = x_data[:n]
    y_data = y_data[:n]

    # perform t-SNE embedding
    vis_data = bh_sne(x_data)

    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.show()


def visualize_features_sklearn(features):
    from time import time
    import matplotlib.pyplot as plt
    from sklearn import manifold

    t0 = time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(features.train_features[0:1000])
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))

    plt.scatter(Y[:, 0], Y[:, 1], c=features.train_labels[0:1000], cmap=plt.cm.Spectral)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    plt.axis('tight')

    plt.show()


if __name__ == "__main__":
    features = Vgg16FeatureDataSet().get_features(overwrite=False)
    # linear_classifier(features)
    # visualize_features(features)
    visualize_features_sklearn(features)
    print("successfully ends")
