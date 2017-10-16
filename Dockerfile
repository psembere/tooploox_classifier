FROM nvidia/cuda:8.0-cudnn5-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
ARG miniconda_version=2-4.2.12

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz openmpi-bin && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda${miniconda_version}-Linux-x86_64.sh && \
    /bin/bash /Miniconda${miniconda_version}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda${miniconda_version}-Linux-x86_64.sh

ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

USER keras

# Python
ARG python_version=2.7

RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install tensorflow-gpu==1.0.0 && \
    conda install Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py && \
    conda install theano pygpu && \
    git clone git://github.com/fchollet/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/fchollet/keras.git && \
    conda clean -yt


ENV PYTHONPATH='/src/:$PYTHONPATH'
ENV HOME /home/keras

WORKDIR $HOME
EXPOSE 8888
CMD jupyter notebook --port=8888 --ip=0.0.0.0

# for hog descriptors
RUN git clone https://github.com/scikit-image/scikit-image.git
WORKDIR $HOME/scikit-image
# for preserving version
RUN git checkout -b my_scirkit_version 88a951f020db3f093c031323eeb8014e02b949a3

RUN pip install -r requirements.txt
RUN conda install Cython
RUN pip install .

#opencv
RUN conda install -c conda-forge opencv

WORKDIR $HOME
# optional cleaning
# RUN rm -rf scikit-image

# for tsne
USER root
RUN apt-get -y install libatlas-base-dev

USER keras
RUN pip install tsne

### repository
ENV REPO_NAME tooploox_classifier
RUN git clone https://github.com/semberecki/$REPO_NAME.git

#SVM libs
RUN git clone https://github.com/ninjin/liblinear
WORKDIR $HOME/liblinear/python
RUN make
WORKDIR $HOME

RUN git clone https://github.com/cjlin1/libsvm.git
WORKDIR $HOME/libsvm/python
RUN make
WORKDIR $HOME

RUN cp liblinear/liblinear.so.1 $REPO_NAME/classifiers/svm_utils
RUN cp libsvm/libsvm.so.2 $REPO_NAME/classifiers/svm_utils

WORKDIR $HOME/$REPO_NAME

RUN python download_dataset.py

RUN pip install seaborn