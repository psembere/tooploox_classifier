from hog_features import HogTrainDataSet, HogTestDataSet
from svm_wrappers import linear_classifier, FeaturesLabelsDataSet
from data_set_deserializer import get_data_set


def get_hog_features(hog_params=None, overwrite=True, visualize=False):
    data_set = get_data_set()
    hog_train_data_set = HogTrainDataSet(data_set, hog_params, visualize)
    hog_test_data_set = HogTestDataSet(data_set, hog_params, visualize)
    if visualize:
        hog_train_data_set.display_picture_with_hog()

    feature_set = FeaturesLabelsDataSet()

    test_features_np, feature_set.test_labels = hog_test_data_set.generate_hog_features(overwrite)
    feature_set.test_features = test_features_np.tolist()
    print("Number of features: " + str(len(feature_set.test_features[0])))

    train_features_list, feature_set.train_labels = hog_train_data_set.generate_hog_features(overwrite)
    feature_set.train_features = train_features_list.tolist()
    return feature_set


if __name__ == "__main__":
    hog_parameters = {
        'orientations': 8,
        'pixels_per_cell': (4, 4),
        'cells_per_block': (2, 2)
    }
    features = get_hog_features(hog_parameters, overwrite=False, visualize=False)

    linear_classifier(features)
    print("successfully ends")
