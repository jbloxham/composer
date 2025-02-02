# MosaicML Composer Docker Images

To simplify environment setup for the MosaicML `Composer` library, we provide a set of Docker Images that users can
leverage.

All Docker images come preinstalled with the following packages:
* Ubuntu 18.04
* Python 3.8.0
* Nvidia CUDA 11.1.1
* PyTorch 1.9.0

Additional dependencies are installed as required by the `composer` library and particular flavor (see below)

## Docker Images

There are three flavors of the `composer` Docker image: 'latest', 'dev' and 'all'.

The specific flavor refers to the specific `pip` packages that are pre-installed and depends on intended usage.

The following table describes the various image flavors:  
| Image  | Dep. Type | Description                                                                                              |
| ------ | --------- | -------------------------------------------------------------------------------------------------------- |
| latest | base      | Minimum `pip` package dependencies installed to enable usage of `composer` library                       |
| dev    | dev       | Package dependencies required for developers.  Use this or 'all' if you plan to contribute to `composer` |
| all    | all       | All package dependencies installed

For details on which specific `pip` packages are installed per flavor, please see the project `setup.py`.

## Pulling Images

Pre-built images can be pulled from [MosaicML's DockerHub Repository](https://hub.docker.com/r/mosaicml/composer):

```bash
# Pull 'base' image
docker pull mosaicml/composer:latest

# Pull 'dev' image
docker pull mosaicml/composer:dev

# Pull 'all' image
docker pull mosaicml/composer:all
```

## Building Images locally

> Note: Docker must be [installed](https://docs.docker.com/get-docker/) on your local machine.

```bash
# Build 'base' image
make

# Build 'dev' image
BUILD_TYPE=dev make

# Build 'all' image
BUILD_TYPE=all make
```
