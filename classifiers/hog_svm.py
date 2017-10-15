from hog_features import HogFeaturesGenerator
from svm_wrappers import linear_classifier

if __name__ == "__main__":
    parameters = {
        'orientations': 8,
        'pixels_per_cell': (4, 4),
        'cells_per_block': (2, 2)
    }
    features = HogFeaturesGenerator(parameters).get_hog_features(overwrite=False, visualize=True)

    linear_classifier(features)
    print("successfully ends")
