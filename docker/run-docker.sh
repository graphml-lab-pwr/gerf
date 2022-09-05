#!/bin/bash

GPU="${1}"

docker run -d \
	   --name "dynamic-gerf-$(whoami)" \
	   -v "${PWD}:/app" \
	   -v "${HOME}/.aws/credentials:/root/.aws/credentials" \
	   --gpus "device=${GPU}" \
	   --ipc=host \
	   dynamic_gerf:latest /bin/bash -c "trap : TERM INT; sleep infinity & wait"
