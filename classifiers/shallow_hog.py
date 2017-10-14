import os

import matplotlib.pyplot as plt
from copy import deepcopy

from skimage.feature import hog
from skimage import data, color, exposure

from utils.data_loader import get_data_set
from utils.globals import PROJECT_PATH
import numpy as np


class HogDataSet(object):
    def __init__(self):
        self.pictures = None
        self.labels = None
        self.labels_text = None

        self.hog_file = None
        self.hog_params = {
            'orientations': 8,
            'pixels_per_cell': (3, 3),
            'cells_per_block': (2, 2)
        }
        self.hog_data = None

    def display_picture_with_hog(self, idx=0):
        image = color.rgb2gray(self.pictures[idx])
        params = deepcopy(self.hog_params)
        params['visualize'] = True
        params['image'] = image
        fd, hog_image = hog(**params)

        self._plot_hog(idx, image, hog_image)

    def _plot_hog(self, idx, image, hog_image):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image ' + self.labels_text[idx])
        ax1.set_adjustable('box-forced')
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        plt.show()

    def generate_hog_features(self, overwrite=False):
        hog_path = os.path.join(PROJECT_PATH, 'data', self.hog_file)
        if os.path.isfile(hog_path) and not overwrite:
            self.hog_data = np.load(hog_path)
        else:
            self.hog_data = self._get_hog_features(self.pictures)
            np.save(hog_path, self.hog_data)
            print(hog_path + ' generated')

        return self.hog_data, self.labels

    @staticmethod
    def _get_hog_features(pictures):
        def get_single_hog(picture):
            image = color.rgb2gray(picture)
            return hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2))

        return [get_single_hog(picture) for picture in pictures]


class HogTrainDataSet(HogDataSet):
    def __init__(self, data_set):
        super(HogTrainDataSet, self).__init__()
        self.pictures = data_set.training_pictures
        self.labels = data_set.training_pictures_labels
        self.labels_text = data_set.get_training_labels_text()
        self.hog_file = 'hog_train_pictures.npy'


class HogTestDataSet(HogDataSet):
    def __init__(self, data_set):
        super(HogTestDataSet, self).__init__()
        self.pictures = data_set.testing_pictures
        self.labels = data_set.testing_pictures_labels
        self.labels_text = data_set.get_testing_labels_text()
        self.hog_file = 'hog_test_pictures.npy'


if __name__ == "__main__":
    # train_hog, train_labels = HogTrainDataSet().generate_hog_features(overwrite=True)
    # test_hog, test_labels = HogTestDataSet().generate_hog_features(overwrite=True)
    data_set = get_data_set()
    hog_train_data_set = HogTrainDataSet(data_set)
    hog_test_data_set = HogTestDataSet(data_set)

    train_hog, train_labels = hog_train_data_set.generate_hog_features(overwrite=True)
    test_hog, test_labels = hog_test_data_set.generate_hog_features(overwrite=True)
    hog_test_data_set.display_picture_with_hog()


    # from liblinearutil import svm_read_problem, train, predict, problem, parameter
    #
    # prob = problem(train_labels, train_hog)
    # param = parameter('-c 4 -B 1')
    # m = train(prob, param)
    # p_label, p_acc, p_val = predict(test_labels, test_hog, m)

    print "kk"
