#!/bin/bash
source $HOME/projects/auto/bin/bash_env.sh

echo ${BASE_CARLA_DOCKER_IMG}
UID=`id -u`
GID=`id -g`

docker run --net host --runtime=nvidia \
	-w /auto \
	-e NVIDIA_VISIBLE_DEVICES=1 \
	-e CUDA_VISIBLE_DEVICES=1 \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	--gpus all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $HOME/projects/auto:/auto \
	-v /usr/local/cuda-11.1:/usr/local/cuda-11.1 \
	-v /usr/share/fonts:/usr/share/fonts \
	-u `id -u`:`id -g`\
	-ti ${BASE_CARLA_DOCKER_IMG} \
	/bin/bash

#	--volume="/etc/group:/etc/group:ro" \
#   	--volume="/etc/passwd:/etc/passwd:ro" \
#	--volume="/etc/shadow:/etc/shadow:ro" \
#	--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \

