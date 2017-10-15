import os

import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from data_set_deserializer import get_data_set
from globals import SERIALIZED_DATA_PATH
import keras


class InceaptionV3FeatureDataSet(object):
    def __init__(self):
        self.model = None

    def _set_model(self):
        self.model = InceptionV3(weights='imagenet', include_top=False)


if __name__ == "__main__":
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
    # keras.callbacks.TensorBoard(log_dir='./logs', write_graph=True)

    # for layers in base_model.layers:
    # print layers.name
    keras.utils.plot_model(base_model, to_file='model.png')

    # base_model = VGG19(weights='imagenet')
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

    a = 4
