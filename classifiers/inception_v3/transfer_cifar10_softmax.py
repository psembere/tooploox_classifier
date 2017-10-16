import numpy as np
import tensorflow as tf

from classifiers.data_augmentation import get_image_generator
from classifiers.inception_v3.download_inception import INCEPTION_PATH
from data_utils import load_CIFAR10
from extract import batch_pool3_features

cifar10_dir = INCEPTION_PATH + '/resources/datasets/cifar-10-batches-py'


def load_pool3_data():
    # Update these file names after you serialize pool_3 values
    X_test_file = 'X_test.npy'
    y_test_file = 'y_test.npy'
    X_train_file = 'X_train.npy'
    y_train_file = 'y_train.npy'
    return np.load(X_train_file), np.load(y_train_file), np.load(X_test_file), np.load(y_test_file)


def serialize_cifar_pool3(X, filename):
    print 'About to generate file: %s' % filename
    sess = tf.InteractiveSession()
    X_pool3 = batch_pool3_features(sess, X)
    np.save(INCEPTION_PATH + "/" + filename, X_pool3)


def serialize_data():
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    serialize_cifar_pool3(X_train, 'X_train')
    serialize_cifar_pool3(X_test, 'X_test')
    np.save(INCEPTION_PATH + '/y_train', y_train)
    np.save(INCEPTION_PATH + '/y_test', y_test)


def serialize_augmented_data(iterations=100, batch_size=100):
    # 1000 x 100 = 10 000 new samples
    image_generator = get_image_generator(batch_size)
    images, labels = image_generator.next()

    print("augmentation", "batch size", 100)
    for idx in xrange(1, iterations):
        new_images, new_labels = image_generator.next()
        images = np.concatenate([images, new_images])
        labels = np.concatenate([labels, new_labels])
        if idx % 10 == 0:
            print(idx)

    serialize_cifar_pool3(images, 'X_train_augmented')
    np.save('y_train_augmented', labels)
