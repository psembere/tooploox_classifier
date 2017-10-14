import liblinearutil
from classifiers.hog_features import HogTrainDataSet, HogTestDataSet
from utils.data_loader import get_data_set


class FeatureLabelsDataSet(object):
    def __init__(self):
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None


def linear_classifier(features):
    prob = liblinearutil.problem(features.train_labels, features.train_features)
    param = liblinearutil.parameter('-c 4 -B 1')
    m = liblinearutil.train(prob, param)
    p_label, p_acc, p_val = liblinearutil.predict(features.test_labels, features.test_features, m)


if __name__ == "__main__":
    data_set = get_data_set()
    hog_train_data_set = HogTrainDataSet(data_set)
    hog_test_data_set = HogTestDataSet(data_set)

    features = FeatureLabelsDataSet()
    overwrite = True
    train_features_list, features.train_labels = hog_train_data_set.generate_hog_features(overwrite)
    test_features_list, features.test_labels = hog_test_data_set.generate_hog_features(overwrite)

    features.train_features = [batch.tolist() for batch in train_features_list]
    features.test_features = [batch.tolist() for batch in test_features_list]

    linear_classifier(features)
    print "kk"
