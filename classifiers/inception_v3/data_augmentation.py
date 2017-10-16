import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from classifiers.data_set_deserializer import get_data_set
from classifiers.inception_v3.download_inception import INCEPTION_PATH
import numpy as np

def get_image_generator():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    data_set = get_data_set()
    PREVIEW_PATH = os.path.join(INCEPTION_PATH, "preview")
    a = np.array(data_set.training_pictures)
    b = np.array(data_set.training_pictures_labels)
    train_generator = datagen.flow(a, b,
                                   save_to_dir=PREVIEW_PATH,
                                   save_format='jpeg',
                                   batch_size=100)
    return train_generator
