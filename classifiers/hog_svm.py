from hog_features import HogFeaturesDataSet
from svm_wrappers import LinearClassifierGenerator

if __name__ == "__main__":
    parameters = {
        'orientations': 8,
        'pixels_per_cell': (4, 4),
        'cells_per_block': (2, 2)
    }
    features = HogFeaturesDataSet(parameters).get_hog_features(overwrite=False,
                                                               visualize=False)
    classifier = LinearClassifierGenerator()
    classifier.classifier_generator(features, params="-s 2 -c 4 -B 1", save=True)
    classifier.evaluate(features)

    print("successfully ends")
