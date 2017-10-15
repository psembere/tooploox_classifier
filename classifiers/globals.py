import os

PROJECT_PATH = os.path.dirname( os.path.dirname(os.path.realpath(__file__)))

EXTRACTED_DATA_SET_DIR_NAME = "cifar-10-batches-py"
EXTRACTED_DATA_SET_PATH = os.path.join(PROJECT_PATH, EXTRACTED_DATA_SET_DIR_NAME)

CHANGE_SERIALIZATION_PATH = False

if CHANGE_SERIALIZATION_PATH:
    SERIALIZED_DATA_PATH = "/Users/piotr/workspace/mac_rl/tooplox_classifier/data"
    print("warning: remote debug path set")
else:
    SERIALIZED_DATA_PATH = os.path.join(PROJECT_PATH, "data")

IMAGE_DIM_X = 32
IMAGE_DIM_Y = 32
RGB_CHANNELS = 3
