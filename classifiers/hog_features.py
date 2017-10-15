import os
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from skimage import color, exposure
from skimage.feature import hog

from globals import SERIALIZED_DATA_PATH


class HogDataSet(object):
    def __init__(self, visualize=False):
        self.pictures = None
        self.labels = None
        if visualize:
            self.labels_text = ""

        self.hog_file = ""
        self.hog_params = dict()
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
        hog_path = os.path.join(SERIALIZED_DATA_PATH, self.hog_file)
        if os.path.isfile(hog_path) and not overwrite:
            self.hog_data = np.load(hog_path)
            print(hog_path + ' loaded')
        else:
            self.hog_data = self._get_hog_features(self.pictures)
            np.save(hog_path, self.hog_data)
            print(hog_path + ' generated')

        return self.hog_data, self.labels

    def _get_hog_features(self, pictures):
        def get_single_hog(pic, hog_params):
            image = color.rgb2gray(pic)
            hog_params['image'] = image
            return hog(**hog_params)

        params = deepcopy(self.hog_params)
        return [get_single_hog(picture, params) for picture in pictures]


class HogTrainDataSet(HogDataSet):
    def __init__(self, data_set, hog_params=None, visualize=False):
        super(HogTrainDataSet, self).__init__(visualize)
        self.pictures = data_set.training_pictures
        self.labels = data_set.training_pictures_labels
        if visualize:
            self.labels_text = data_set.get_training_labels_text()
        self.hog_file = 'hog_train_pictures.npy'
        self.hog_params = hog_params if hog_params else dict()


class HogTestDataSet(HogDataSet):
    def __init__(self, data_set, hog_params=None, visualize=False):
        super(HogTestDataSet, self).__init__()
        self.pictures = data_set.testing_pictures
        self.labels = data_set.testing_pictures_labels
        if visualize:
            self.labels_text = data_set.get_testing_labels_text()
        self.hog_file = 'hog_test_pictures.npy'
        self.hog_params = hog_params if hog_params else dict()
