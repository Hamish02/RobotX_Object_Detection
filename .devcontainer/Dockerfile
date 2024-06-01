# FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04
# FROM stereolabs/zed:4.1-devel-cuda12.1-ubuntu22.04
# FROM stereolabs/zed:4.1-devel-cuda12.1-ubuntu20.04
FROM stereolabs/zed:4.0-gl-devel-cuda11.4-ubuntu20.04

# # Set noninteractive installation mode
# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y python3-requests
# RUN apt-get install -y git
# RUN apt-get install libglib2.0-0
# # RUN python3 get_python_api.py
# RUN pip install ultralytics
# RUN pip install PyOpenGL
# RUN python3 get_python_api.py
# RUN apt-get update


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    # wget \
    # unzip \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python and pip
RUN apt-get update && apt-get install -y \
    xvfb \
    freeglut3-dev \
    libgl1-mesa-glx \
    libegl1-mesa \
    python3-opengl \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get install -y freeglut3

# Install Python dependencies for object detection
RUN pip3 install \
    numpy==1.24.4 \
    opencv-python \
    torch==2.2.2 \
    torchvision==0.17.2 \
    cython \
    pyopengl \
    ultralytics==8.2.18 \
    torchaudio==2.2.2 \
    # libgl1 \
    # libglib2.0-0\
    # libpython3-dev \
    gnupg \
    # g++ \
    # pytorch-lightning \
    && rm -rf /root/.cache/pip

RUN python3 /usr/local/zed/get_python_api.py

# RUN apt-get install xvfb
# RUN export DISPLAY=:99
# RUN Xvfb :99 -screen 0 1024x768x24 &