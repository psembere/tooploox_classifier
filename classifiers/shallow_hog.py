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


class HogDataSet(object):
    def __init__(self):
        self.data_set = get_data_set()
        self.hog_file = None

    def generate_hog_features(self, overwrite=False):
        hog_path = os.path.join(PROJECT_PATH, 'data', self.hog_file)
        if os.path.isfile(hog_path) and not overwrite:
            hog_data = np.load(hog_path).tolist()
        else:
            pictures = self._get_pictures()
            hog_data = self._get_hog_features(pictures)
            np.save(hog_path, hog_data)
            print(hog_path + ' generated')

        return hog_data, self._get_labels()

    def _get_pictures(self):
        pass

    def _get_labels(self):
        pass

    @staticmethod
    def _get_hog_features(pictures):
        def get_single_hog(picture):
            image = color.rgb2gray(picture)
            return hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2))

        return [get_single_hog(picture) for picture in pictures]


class HogTrainDataSet(HogDataSet):
    def __init__(self):
        super(HogTrainDataSet, self).__init__()
        self.hog_file = 'hog_train_pictures.npy'

    def _get_labels(self):
        return self.data_set.get_training_labels_indexes()

    def _get_pictures(self):
        return self.data_set.get_training_pictures()


class HogTestDataSet(HogDataSet):
    def __init__(self):
        super(HogTestDataSet, self).__init__()
        self.hog_file = 'hog_train_pictures.npy'

    def _get_labels(self):
        return self.data_set.get_testing_labels_indexes()

    def _get_pictures(self):
        return self.data_set.get_testing_labels_indexes()


if __name__ == "__main__":
    train_hog, train_labels = HogTrainDataSet().generate_hog_features(overwrite=True)
    test_hog, test_labels = HogTrainDataSet().generate_hog_features(overwrite=True)

    from liblinearutil import svm_read_problem, train, predict, problem, parameter

    prob = problem(train_labels, train_hog)
    param = parameter('-c 4 -B 1')
    m = train(prob, param)
    p_label, p_acc, p_val = predict(test_labels, test_hog, m)

    print "kk"
