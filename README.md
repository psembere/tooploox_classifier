# tooplox_classifier

for creating docker container with required versions of software to develop classifiers run: <br /> <br />
 __make__
<br />
<br />
This it creates and runs jupyter notebook, which allow to develop and run this code 

This repostitory contains three approaches to classification based on transfer learning <br />

 - Hog descriptor from scikit-image v0.14dev + SVM
 - Vgg16  network without last layer from Keras with Tensorflow backed + SVM
 - Inception V3 with features from pool3:0 layer from Tensorflow + SVM
 
 
 The SVM are used from liblinear and libsvm libraries with Python wrappers