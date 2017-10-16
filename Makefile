DATA?="${HOME}/keras_data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
SRC=$(shell dirname `pwd`)
IMAGE_NAME=keras_container

all: notebook

build:
	docker build -t $(IMAGE_NAME) -f $(DOCKER_FILE) .

notebook: build
	$(DOCKER) run -it --net=host -v $(DATA):/home/keras/data  $(IMAGE_NAME)

bash: build
	$(DOCKER) run -it --net=host -v $(DATA):/home/keras/data $(IMAGE_NAME) bash