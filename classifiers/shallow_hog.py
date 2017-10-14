import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure

from utils.data_loader import get_data_set


def display_picture_with_hog(picture):
    image = color.rgb2gray(picture)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2), visualize=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()


def get_hog_features(pictures):
    def get_single_hog(picture):
        image = color.rgb2gray(picture)
        return hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2))

    return [get_single_hog(picture) for picture in pictures]



if __name__ == "__main__":
    data_set = get_data_set()
    pictures = data_set.get_training_pictures()

    pics =  data_set.get_test_picutres()
    labels =  data_set.get_test_labels()
    # display_picture_with_hog(pictures[0])
    # hog_features = get_hog_features(pictures)
    # from liblinearutil import *
    #
    # y, x = svm_read_problem('../heart_scale')
    print "kk"
