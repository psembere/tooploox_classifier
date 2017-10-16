from classifiers.svm_wrappers import LinearClassifierGenerator, KernelClassifierGenerator
from classifiers.t_sne import visualize_features_pypl_tsne_fast
from vgg16_features import Vgg16FeatureDataSet

if __name__ == "__main__":
    features = Vgg16FeatureDataSet().get_features(overwrite=False)
    #visualize_features_pypl_tsne_fast(features, 1000)

    print "classifier"
    linear_classifier = LinearClassifierGenerator()
    linear_classifier.classifier_generator(features, params="-s 2 -c 4 -B 1", save=True)

    # kernel_classifier = KernelClassifierGenerator()
    # kernel_classifier.classifier_generator(features, params="-h 0", save=True)
    # classifier.evaluate(features)
    print("successfully ends")
