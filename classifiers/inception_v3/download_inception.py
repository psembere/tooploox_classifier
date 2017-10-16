import distutils
import os
import shutil

from classifiers.globals import EXTRACTED_DATA_SET_PATH, PROJECT_PATH
from download_utils import download_data_set, extract


def prepare_inception():
    INCEPTION_V3_FILENAME = "inception-2015-12-05.tgz"
    INCEPTION_V3_URL = "http://download.tensorflow.org/models/image/imagenet/" + INCEPTION_V3_FILENAME
    print ("inception data set downloading")
    download_data_set(INCEPTION_V3_URL, INCEPTION_V3_FILENAME)

    INCEPTION_PATH = os.path.join(PROJECT_PATH, "classifiers", "inception_v3")
    RESOURCES_PATH = os.path.join(INCEPTION_PATH, "resources")
    os.mkdir(RESOURCES_PATH)
    DOWNLOADED_INCEPTION = os.path.join(INCEPTION_PATH, INCEPTION_V3_FILENAME)
    MOVED_INCEPTION = os.path.join(RESOURCES_PATH, INCEPTION_V3_FILENAME)

    shutil.move(DOWNLOADED_INCEPTION, MOVED_INCEPTION)
    extract(MOVED_INCEPTION)
    #os.remove(MOVED_INCEPTION)

    DATA_SETS_PATH = os.path.join(RESOURCES_PATH, "datasets")
    os.mkdir(DATA_SETS_PATH)
    distutils.dir_util.copy_tree(EXTRACTED_DATA_SET_PATH, DATA_SETS_PATH)