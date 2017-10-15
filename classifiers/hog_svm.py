from hog_features import HogFeaturesDataSet
from svm_wrappers import linear_classifier_generate, linear_classifier_evaluate

if __name__ == "__main__":
    parameters = {
        'orientations': 8,
        'pixels_per_cell': (4, 4),
        'cells_per_block': (1, 1)
    }
    features = HogFeaturesDataSet(parameters).get_hog_features(overwrite=False,
                                                               visualize=False)
    linear_classifier_generate(features, params="-c 4 -B 1")
    linear_classifier_evaluate(features)

    # kernel_classifier(features)
    print("successfully ends")
