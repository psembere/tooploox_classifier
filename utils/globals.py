import os

UTILS_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = os.path.dirname(UTILS_DIR_PATH)

DATA_SET_NAME = "cifar-10-batches-py"
DATA_SET_PATH = os.path.join(UTILS_DIR_PATH, "..", DATA_SET_NAME)

IMAGE_DIM_X = 32
IMAGE_DIM_Y = 32
RGB_CHANNELS = 3