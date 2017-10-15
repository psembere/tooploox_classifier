import os

import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from data_set_deserializer import get_data_set
from globals import SERIALIZED_DATA_PATH
from svm_wrappers import FeaturesDataSet, linear_classifier


class InceaptionV3FeatureDataSet(object):
    def __init__(self):
        self.model = None

    def _set_model(self):
        self.model = InceptionV3(weights='imagenet', include_top=False)


if __name__ == "__main__":
    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
