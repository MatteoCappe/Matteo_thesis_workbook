# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

ARG CODE_DIR=/usr/local/src

RUN apt update

# install python3.7
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.7 python3.7-dev python3-pip python3.7-distutils
RUN ln -sfn /usr/bin/python3.7 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

# basic dependencies
RUN apt install -y \
    wget \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    python3-dev \
    python3-distutils \
    python3-pip \
    libopencv-dev

# Upgrade pip and install PyTorch and torchvision
RUN pip install torch torchvision torchaudio

# Install the other dependencies via pip
RUN pip install --no-cache-dir numpy ipython matplotlib tqdm pylint jupyterlab scipy numba ipdb yacs timm
    
# Clone and install SuperPointPretrainedNetwork
RUN git clone https://github.com/magicleap/SuperPointPretrainedNetwork.git /opt/SuperPointPretrainedNetwork && \
    cd /opt/SuperPointPretrainedNetwork

# Clone and install SuperGluePretrainedNetwork
RUN git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git /opt/SuperGluePretrainedNetwork && \
    cd /opt/SuperGluePretrainedNetwork

# version specifics for OnePose
RUN pip install --no-cache-dir h5py loguru albumentations pytorch-lightning aiohttp aioredis pydegensac opencv-python opencv-python-headless einops kornia pickle5 hydra-core omegaconf pycocotools wandb rich transforms3d natsort

# ==3.1.0 ==0.5.3==0.5.1 ==1.5.10 ==3.7 ==1.3.1 ==0.1.2 ==4.1.2.30 ==4.4.0.46 ==0.3.0 ==0.4.1 ==0.0.11 ==1.1.1 ==2.1.2 # ==2.0.4 ==0.12.17 ==12.4.4 ==0.3.1 ==8.1.0

# Clone and install OnePose
RUN git clone https://github.com/zju3dv/OnePose.git /opt/OnePose && \
    cd /opt/OnePose 