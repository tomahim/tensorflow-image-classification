# Tensorflow image classification experimentations

This project aims to share some image classification experimentation.

### Tensorflow installation

I recommend to install Tensorflow by using [Anaconda](https://conda.io/docs/user-guide/install/index.html#) and create a specific [Anaconda environnement](https://conda.io/docs/user-guide/tasks/manage-environments.html) for Tensorflow as it's describe here :

https://www.tensorflow.org/install/install_windows#id=installing_with_anaconda (example for windows)

#### Pre-requirements for using GPU with Tensorflow

If you have a Nvidia graphic card, you can use it to speed up some image processing while running deep learning algorithms. 

Tensorflow works well with the Nvidia CUDA Toolkit, you will need to install the following tools in this order :

- [Microsoft Visual Studio](https://www.visualstudio.com/fr/) : If you're on Windows, it's needed for better integration with CUDA
- [CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-toolkit) : be carefull to install the CUDA 9.0 version as it's for now the last release Tensorflow supports
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) : The GPU Accelerated Deep Learning library by Nvidia

