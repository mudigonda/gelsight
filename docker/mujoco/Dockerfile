FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

LABEL maintainer="Roberto Calandra <roberto.calandra@berkeley.edu>"

# Install pip
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libglu1-mesa \
    libxi-dev \
    libxmu-dev \
    libglu1-mesa-dev \
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
    python3-tk \
    python3-dev \
    net-tools \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install basic computational libraries
RUN pip3 install --upgrade pip
RUN pip3 install matplotlib scipy scikit-learn future
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
#RUN pip3 install numpy tensorflow-gpu==1.4
#RUN pip3 install GPy
#RUN pip3 install cma

# Install MuJoCo
RUN pip3 install gym==0.9.4
RUN mkdir -p /.mujoco && cd /.mujoco && wget https://www.roboti.us/download/mjpro131_linux.zip && unzip mjpro131_linux.zip
ENV MUJOCO_PY_MJKEY_PATH="/.mujoco/mjkey.txt"
ENV MUJOCO_PY_MJPRO_PATH="/.mujoco/mjpro131"
RUN pip3 install mujoco-py==0.5.7

# Install additional requirements for RL package
RUN pip3 install datetime gitpython h5py tqdm dotmap cython scipyplot torchvision

# GPFlow
#RUN git clone https://github.com/GPflow/GPflow.git
#RUN pip3 install pandas multipledispatch pytest
#RUN cd GPflow/ && pip install . --no-deps

# Set matplotlib defaults
RUN export MATPLOTLIBRC="/"
RUN echo "backend: PDF" > /matplotlibrc
RUN alias python=python3

# No screen
RUN apt-get update && apt-get install -y \
    xvfb

# Environment setup
RUN echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> /root/.bashrc
RUN echo 'alias python=python3' >> /root/.bashrc
CMD /bin/bash