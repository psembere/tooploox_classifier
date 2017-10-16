import os

from classifiers.inception_v3.download_inception import INCEPTION_PATH
from classifiers.svm_wrappers import LinearClassifierGenerator, KernelClassifierGenerator, FeaturesDataSet
from classifiers.t_sne import visualize_features_pypl_tsne_fast
import numpy as np


def load_cifar_features():
    features = FeaturesDataSet()
    features.train_features = np.load(os.path.join(INCEPTION_PATH, "X_train.npy")).tolist()
    features.train_labels = np.load(os.path.join(INCEPTION_PATH, "y_train.npy")).tolist()

    features.test_features = np.load(os.path.join(INCEPTION_PATH, "X_test.npy")).tolist()
    features.test_labels = np.load(os.path.join(INCEPTION_PATH, "y_test.npy")).tolist()

    return features


def add_augmented_features(features):
    train_augmented_features = np.load(os.path.join(INCEPTION_PATH, "X_train_augmented.npy")).tolist()
    train_augmented_labels = np.load(os.path.join(INCEPTION_PATH, "y_train_augmented.npy")).tolist()

    features.train_features += train_augmented_features
    features.train_labels += train_augmented_labels

if __name__ == "__main__":
    features = load_cifar_features()
    add_augmented_features(features)

    # visualize_features_pypl_tsne_fast(features, 10000)

    linear_classifier = LinearClassifierGenerator()
    kernel_classifier = KernelClassifierGenerator()

    linear_classifier.classifier_generator(features, params="", save=True)  # 87, 88.43 with augmentation
    # kernel_classifier.classifier_generator(features, params="", save=True)
