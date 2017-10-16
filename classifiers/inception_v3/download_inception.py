import os
import shutil

from classifiers.globals import EXTRACTED_DATA_SET_PATH, PROJECT_PATH, EXTRACTED_DATA_SET_DIR_NAME
from download_utils import download_data_set, extract_tar

INCEPTION_V3_FILENAME = "inception-2015-12-05.tgz"
INCEPTION_V3_URL = "http://download.tensorflow.org/models/image/imagenet/" + INCEPTION_V3_FILENAME

INCEPTION_PATH = os.path.join(PROJECT_PATH, "classifiers", "inception_v3")
INCEPTION_RESOURCES_PATH = os.path.join(INCEPTION_PATH, "resources")

DOWNLOADED_INCEPTION = os.path.join(INCEPTION_PATH, INCEPTION_V3_FILENAME)
MOVED_INCEPTION = os.path.join(INCEPTION_RESOURCES_PATH, INCEPTION_V3_FILENAME)

DATA_SETS_PATH = os.path.join(INCEPTION_RESOURCES_PATH, "datasets")

CIFAR_COPY = os.path.join(DATA_SETS_PATH, EXTRACTED_DATA_SET_DIR_NAME)


def prepare_inception():
    print ("inception data set downloading")
    download_data_set(INCEPTION_V3_URL, DOWNLOADED_INCEPTION)
    print ("inception data downloaded")

    os.mkdir(INCEPTION_RESOURCES_PATH)

    shutil.move(DOWNLOADED_INCEPTION, MOVED_INCEPTION)
    extract_tar(MOVED_INCEPTION, INCEPTION_RESOURCES_PATH)
    os.remove(MOVED_INCEPTION)

    os.mkdir(DATA_SETS_PATH)
    shutil.copytree(EXTRACTED_DATA_SET_PATH, CIFAR_COPY)


def generate_inception_features():
    serialized_data_path = os.path.join(INCEPTION_PATH, "X_train.npy")
    if not os.path.isfile(serialized_data_path):
        print("generate")
        execfile(os.path.join(INCEPTION_PATH, "transfer_cifar10_softmax.py"))
    else:
        print("features generated")
