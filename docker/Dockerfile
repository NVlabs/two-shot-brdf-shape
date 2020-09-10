# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.
# Adapted from:
# https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile
# Build with:
# docker build -t nvidia-twoshot:v1 -f Dockerfile ..

ARG UBUNTU_VERSION=16.04

ARG ARCH=
ARG CUDA=10.0
FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=7.4.1.5-1

# Needed for string substitution 
SHELL ["/bin/bash", "-c"]
# Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    software-properties-common \
    cuda-command-line-tools-${CUDA/./-} \
    cuda-cublas-${CUDA/./-} \
    cuda-cufft-${CUDA/./-} \
    cuda-curand-${CUDA/./-} \
    cuda-cusolver-${CUDA/./-} \
    cuda-cusparse-${CUDA/./-} \
    curl \
    libcudnn7=${CUDNN}+cuda${CUDA} \
    libcap-dev \
    libopenexr-dev \
    zlib1g-dev
# For CUDA profiling, TensorFlow requires CUPTI.
# Additionally, add CUDA itself to the path.
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda

ARG USE_PYTHON_3_NOT_2=True
ARG PY_VER=3.6
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1

# Setup python
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python${PY_VER} python${PY_VER}-dev python${PY_VER}-tk && \
    curl --silent https://bootstrap.pypa.io/get-pip.py | python${PY_VER}

# Backwards compatility and make sure pipenv/pip are working
RUN rm -fr /usr/bin/python3 && \
    ln /usr/bin/python${PY_VER} /usr/bin/python3 && \
    ln /usr/bin/python${PY_VER} /usr/bin/python && \
    ln /usr/local/bin/pip /usr/bin/pip

# Backwards compatility and make sure pipenv/pip are working
RUN pip install --no-cache-dir -U pipenv

RUN set -ex && mkdir /deps

# Only copy the pipfiles for now so that everything stays cached
COPY Pipfile* /deps/
RUN cd /deps && pipenv lock -r >> /deps/requirements.txt && \
    pip install --no-cache-dir -U -r /deps/requirements.txt && \
    rm -rf /deps

COPY . /net/
WORKDIR /net