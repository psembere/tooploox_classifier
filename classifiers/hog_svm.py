from hog_features import HogTrainDataSet, HogTestDataSet
from svm_wrappers import linear_classifier, FeaturesDataSet
from data_set_deserializer import get_data_set


class HogFeaturesGenerator(object):
    def __init__(self, hog_parameters=None):
        self.hog_parameters = hog_parameters

    def get_hog_features(self, overwrite=True, visualize=False):
        data_set = get_data_set()
        hog_train_data_set = HogTrainDataSet(data_set, self.hog_parameters, visualize)
        hog_test_data_set = HogTestDataSet(data_set, self.hog_parameters, visualize)
        if visualize:
            hog_train_data_set.display_picture_with_hog()

        feature_set = FeaturesDataSet()
        self._set_train_features(feature_set, hog_train_data_set, overwrite)
        self._set_test_features(feature_set, hog_test_data_set, overwrite)

        print("Number of features: " + str(len(feature_set.train_features[0])))

        return feature_set

    @staticmethod
    def _set_train_features(feature_set, hog_train_data_set, overwrite):
        train_features_list, feature_set.train_labels = hog_train_data_set.generate_hog_features(overwrite)
        feature_set.train_features = train_features_list.tolist()

    @staticmethod
    def _set_test_features(feature_set, hog_test_data_set, overwrite):
        test_features_np, feature_set.test_labels = hog_test_data_set.generate_hog_features(overwrite)
        feature_set.test_features = test_features_np.tolist()


if __name__ == "__main__":
    parameters = {
        'orientations': 8,
        'pixels_per_cell': (4, 4),
        'cells_per_block': (2, 2)
    }
    features = HogFeaturesGenerator(parameters).get_hog_features(overwrite=False, visualize=True)

    linear_classifier(features)
    print("successfully ends")
