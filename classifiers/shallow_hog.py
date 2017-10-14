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


def set_files_existance(file_path):
    for key in file_path:
        if os.path.isfile(key):
            file_path[key] = False



def generate_hog_train_features():
    file_name_train_hog = os.path.join(PROJECT_PATH, 'data', 'hog_train.npy')
    file_name_train_labels = os.path.join(PROJECT_PATH, 'data', 'hog_train.npy')
    generate_files = {
        os.path.join(PROJECT_PATH, 'data', 'hog_train.npy'): True,
        os.path.join(PROJECT_PATH, 'data', 'hog_train.npy'): True
    }
    if os.path.isfile(file_name_train_hog):
        hog_train = np.load(file_name_train_hog)
    else:
        data_set = get_data_set()
        pictures = data_set.get_training_pictures()
        hog_train = get_hog_features(pictures)
        np.save(file_name_train_hog, hog_train)
        print("hog train generated")

    if os.path.isfile(file_name_train_hog):
        hog_labels = np.load(file_name_train_labels)
    else:
        data_set = get_data_set()
        hog_labels = data_set.get_training_labels_indexes()
        np.save(file_name_train_labels, hog_labels)

    return hog_train, hog_labels


if __name__ == "__main__":
    hog_train, hog_labels = generate_hog_train_features()

    from liblinearutil import svm_read_problem, train, predict, problem, parameter

    prob = problem(hog_labels, hog_train)
    param = parameter('-c 4 -B 1')
    m = train(prob, param)
    # p_label, p_acc, p_val = predict(data_set.get_test_labels_indexes(), hog_test, m)




    print "kk"
