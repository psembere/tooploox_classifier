import os

from classifiers.inception_v3.download_inception import INCEPTION_PATH
from classifiers.svm_wrappers import LinearClassifierGenerator, KernelClassifierGenerator, FeaturesDataSet
from classifiers.t_sne import visualize_features_pypl_tsne_fast
import numpy as np


def load_cifar_features():
    features = FeaturesDataSet()
    features.train_features = np.load(os.path.join(INCEPTION_PATH, "X_train.npy")).tolist()
    features.train_labels =  np.load(os.path.join(INCEPTION_PATH, "y_train.npy")).tolist()

    features.test_features =  np.load(os.path.join(INCEPTION_PATH, "X_test.npy")).tolist()
    features.test_labels =  np.load(os.path.join(INCEPTION_PATH, "y_test.npy")).tolist()

    return features


if __name__ == "__main__":
    features = load_cifar_features()

    # visualize_features_pypl_tsne_fast(features, 10000)

    classifier = LinearClassifierGenerator()
    classifier.classifier_generator(features, params="", save=True)
