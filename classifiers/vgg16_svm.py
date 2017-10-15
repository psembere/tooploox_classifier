from t_sne import visualize_features_pypl_tsne_fast
from vgg16_features import Vgg16FeatureDataSet

if __name__ == "__main__":
    features = Vgg16FeatureDataSet().get_features(overwrite=False)
    # linear_classifier(features)
    visualize_features_pypl_tsne_fast(features, 50000)
    print("successfully ends")
