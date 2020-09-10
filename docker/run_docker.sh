#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

CONTAINER_NAME=$1
if [[ -z "${CONTAINER_NAME}" ]]; then
    CONTAINER_NAME=nvidia-twoshot-v1
fi

echo "Container name     : ${CONTAINER_NAME}"
echo "Host directory     : ${HOST_DIR}"
echo "Container directory: ${CONTAINER_DIR}"
CONTAINER_ID=`docker ps -aqf "name=^/${CONTAINER_NAME}$"`
if [ -z "${CONTAINER_ID}" ]; then
    echo "Creating new docker container."
    xhost +local:root
    docker run --gpus all  -it --privileged --network=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw --env="DISPLAY" --name=${CONTAINER_NAME} nvidia-twoshot:v1 bash
else
    echo "Found an existing docker container: ${CONTAINER_ID}."
    # Check if the container is already running and start if necessary.
    if [ -z `docker ps -qf "name=^/${CONTAINER_NAME}$"` ]; then
        xhost +local:${CONTAINER_ID}
        echo "Starting and attaching to ${CONTAINER_NAME} container..."
        docker start ${CONTAINER_ID}
        docker attach ${CONTAINER_ID}
    else
        echo "Found running ${CONTAINER_NAME} container, attaching bash..."
        docker exec -it ${CONTAINER_ID} bash
    fi
fi
