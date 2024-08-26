# Object Detection web UI

A web interface for Object Detection, implemented using Gradio library.

## Installation and Running

### Installation on Windows with Conda

1. install env

    ```bash
    git clone https://github.com/tensorflow/models.git
    conda create -n od python=3.9
    conda activate od
    cd models/research
    conda install protobuf
    protoc object_detection/protos/*.proto --python_out=.
    copy object_detection\packages\tf2\setup.py .
    # pip install protobuf==3.20
    python -m pip install .
    python object_detection/builders/model_builder_tf2_test.py
    pip install gradio
    ```
    
2. run server
    
    ```bash
    python main.py
    ```
    

## Installation on Windows with Docker

1. Clone
    
    ```bash
    git clone https://github.com/tensorflow/models.git
    ```
    
2. update   `research/object_detection/dockerfiles/tf2/Dockerfile`
    
    ```docker
    # FROM tensorflow/tensorflow:2.2.0-gpu
    FROM tensorflow/tensorflow:2.10.1-gpu

    ARG DEBIAN_FRONTEND=noninteractive

    RUN apt-key del 7fa2af80
    RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

    # Install apt dependencies
    RUN apt-get update && apt-get install -y \
        git \
        gpg-agent \
        python3-cairocffi \
        protobuf-compiler \
        python3-pil \
        python3-lxml \
        python3-tk \
        wget

    # Install gcloud and gsutil commands
    # https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
    # RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    #     echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    #     curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    #     apt-get update -y && apt-get install google-cloud-sdk -y

    # Add new user to avoid running as root
    RUN useradd -ms /bin/bash tensorflow
    USER tensorflow
    WORKDIR /home/tensorflow

    # Copy this version of of the model garden into the image
    COPY --chown=tensorflow . /home/tensorflow/models

    # Compile protobuf configs
    RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
    WORKDIR /home/tensorflow/models/research/

    RUN cp object_detection/packages/tf2/setup.py ./
    ENV PATH="/home/tensorflow/.local/bin:${PATH}"

    RUN python -m pip install -U pip
    RUN python -m pip install .

    ENV TF_CPP_MIN_LOG_LEVEL 3
    ```
    
3. Build Image
    
    ```bash
    docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t object-detection .
    ```
    
4. Build Image
    
    ```bash
    docker build -t object-detection-webui .
    ```
    
5. run sever
    
    ```bash
    docker run --gpus all -p 7860:7860 -v ./projects:/data/projects -v ./datasets:/data/datasets  object-detection-webui
    ```