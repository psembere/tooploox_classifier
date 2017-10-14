import liblinearutil
from classifiers.hog_features import HogTrainDataSet, HogTestDataSet
from utils.data_loader import get_data_set


class FeatureLabelsDataSet(object):
    def __init__(self):
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None


def linear_classifier(feature_set):
    prob = liblinearutil.problem(feature_set.train_labels, feature_set.train_features)
    param = liblinearutil.parameter('-c 4 -B 1')
    m = liblinearutil.train(prob, param)
    p_label, p_acc, p_val = liblinearutil.predict(feature_set.test_labels, feature_set.test_features, m)


def get_hog_features(overwrite=False, visualize=False):
    data_set = get_data_set()
    hog_train_data_set = HogTrainDataSet(data_set)
    hog_test_data_set = HogTestDataSet(data_set)
    if visualize:
        hog_train_data_set.display_picture_with_hog()

    feature_set = FeatureLabelsDataSet()

    train_features_list, feature_set.train_labels = hog_train_data_set.generate_hog_features(overwrite)
    test_features_list, feature_set.test_labels = hog_test_data_set.generate_hog_features(overwrite)
    feature_set.train_features = [batch.tolist() for batch in train_features_list]
    feature_set.test_features = [batch.tolist() for batch in test_features_list]

    return feature_set


if __name__ == "__main__":
    features = get_hog_features()

    linear_classifier(features)
    print "kk"
