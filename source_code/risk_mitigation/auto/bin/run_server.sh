#!/bin/bash
source $HOME/projects/auto/bin/bash_env.sh

echo ${BASE_CARLA_DOCKER_IMG}
docker run --net host --runtime=nvidia \
	-w /auto \
	-e NVIDIA_VISIBLE_DEVICES=0 \
	-e DISPLAY=$DISPLAY \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $HOME/projects/auto:/auto \
	-ti ${BASE_CARLA_DOCKER_IMG} \
	/bin/bash /auto/sim/carla-0.9.10/CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl


#/Game/Maps/Town01 -opengl -carla-server -benchmark -fps=10 -windowed -ResX=800 -ResY=600 
