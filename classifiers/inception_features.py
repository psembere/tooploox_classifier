import os

from classifiers.inception_v3.download_inception import INCEPTION_PATH
from classifiers.inception_v3.transfer_cifar10_softmax import serialize_data, serialize_data_with_augmentation


def generate_inception_features(generate_augmentation=True):
    serialized_data_path = os.path.join(INCEPTION_PATH, "X_train.npy")
    if not os.path.isfile(serialized_data_path):
        print("generate")
        # execfile(os.path.join(INCEPTION_PATH, "transfer_cifar10_softmax.py"))
        serialize_data()
    else:
        print("features generated")

    if generate_augmentation:
        serialize_data_with_augmentation()
