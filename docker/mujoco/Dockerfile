# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
FROM nvidia/cuda:8.0-devel-ubuntu16.04

LABEL maintainer="Roberto Calandra <roberto.calandra@berkeley.edu>"

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libc6 \
    libc6-dev \
    python3 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-dev \
    net-tools \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

# Install MuJoCo
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH
# Upgrade pip3
RUN pip3 install --upgrade pip
# MuJoCo-py
RUN pip3 install -U 'mujoco-py<1.50.2,>=1.50.1'
# Install OpenAI Gym
RUN pip3 install gym pyopengl
# Install Pytorch
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision
