from hog_features import HogFeaturesDataSet
from svm_wrappers import linear_classifier, kernal_classifier

if __name__ == "__main__":
    parameters = {
        'orientations': 8,
        'pixels_per_cell': (4, 4),
        'cells_per_block': (1, 1)
    }
    features = HogFeaturesDataSet(parameters).get_hog_features(overwrite=False,
                                                               visualize=False)

    #linear_classifier(features)
    kernal_classifier(features)
    print("successfully ends")
