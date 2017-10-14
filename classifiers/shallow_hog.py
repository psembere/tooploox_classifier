import os

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure

from utils.data_loader import get_data_set
from utils.globals import PROJECT_PATH
import numpy as np


def display_picture_with_hog(picture):
    image = color.rgb2gray(picture)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2), visualize=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()


def get_hog_features(pictures):
    def get_single_hog(picture):
        image = color.rgb2gray(picture)
        return hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2))

    return [get_single_hog(picture) for picture in pictures]



def generate_hog_train_features():
    file_name_train_hog = os.path.join(PROJECT_PATH, 'data', 'hog_train_pictures.npy')
    data_set = get_data_set()
    if os.path.isfile(file_name_train_hog):
        hog_train = np.load(file_name_train_hog)
    else:
        pictures = data_set.get_training_pictures()
        hog_train = get_hog_features(pictures)
        np.save(file_name_train_hog, hog_train)
        print(file_name_train_hog + ' generated')

    return hog_train.tolist(), data_set.get_training_labels_indexes()


def generate_hog_test_features():
    file_name_test_hog = os.path.join(PROJECT_PATH, 'data', 'hog_test_pictures.npy')
    data_set = get_data_set()
    if os.path.isfile(file_name_test_hog):
        hog_train = np.load(file_name_test_hog)
    else:
        pictures = data_set.get_testing_pictures()
        hog_train = get_hog_features(pictures)
        np.save(file_name_test_hog, hog_train)
        print(file_name_test_hog + ' generated')

    return hog_train.tolist(), data_set.get_testing_labels_indexes()

if __name__ == "__main__":
    data_set = get_data_set()
    train_hog, train_labels = generate_hog_train_features()
    test_hog, test_labels = generate_hog_test_features()

    from liblinearutil import svm_read_problem, train, predict, problem, parameter

    prob = problem(train_labels, train_hog)
    param = parameter('-c 4 -B 1')
    m = train(prob, param)
    p_label, p_acc, p_val = predict(test_labels, test_hog, m)


    print "kk"
