
## Requirements:

The code is written to be run on Python 3.5.  
The simulator is using MuJoCo 1.31 and/or Flex 2.  
Most of the code is written to run using docker and/or [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)


## Compile Docker Image


### Flex
Make sure that you have a copy of FlexRobotics-master.zip inside the /docker/flex/ folder.
Then run
```
cd docker/flex
docker build -t pytorchflex $(pwd)/docker/flex/.
```

### MuJoCo
Make sure that you have a valid mjkey.txt inside the /docker/mujoco/ folder. Then run
```
cd docker/mujoco
docker build -t pytorchmujoco $(pwd)/docker/mujoco/.
```

## Run Docker
`docker run --runtime=nvidia --rm pytorchflex nvidia-smi`

To run the docker image:
`docker run --runtime=nvidia --rm -ti -v  $(pwd):/home/code -v $(pwd)/docker/mujoco/mjkey.txt:/.mujoco/mjkey.txt --ipc=host pytorchmujoco`   
`docker run --runtime=nvidia --rm -e DISPLAY=$DISPLAY -ti -v  $(pwd):/home/code -v $(pwd)/docker/mujoco/mjkey.txt:/.mujoco/mjkey.txt --ipc=host pytorchmujoco`





## Run demo:

demo_mujoco_pointmass