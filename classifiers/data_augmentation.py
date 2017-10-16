import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from classifiers.data_set_deserializer import get_data_set
from classifiers.inception_v3.download_inception import INCEPTION_PATH
import numpy as np

# PREVIEW_PATH = os.path.join(INCEPTION_PATH, "preview")


def get_image_generator(batch_size=100):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    data_set = get_data_set()

    a = np.array(data_set.training_pictures)
    b = np.array(data_set.training_pictures_labels)
    train_generator = datagen.flow(a, b,
                                   save_format='jpeg',
                                   batch_size=batch_size)
    # optional save dir save_to_dir=PREVIEW_PATH,
    return train_generator
