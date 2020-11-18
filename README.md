# trax_experiments
google trax experiments on attention-based architectures   


## Installing google trax on AWS (px.*)

Requires tensorflow 2 >~ 2.3 (to be installed on top of the cuda drivers on the machine). 

Requires installing google jax for CPU: https://github.com/google/jax#installation. 

Get the version of the existing CUDA installation you want to use: 

```sh
>> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```

Then: 

```sh
pip install --upgrade jax jaxlib==0.1.57+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

with cuda110 for CUDA 11.0, cuda102 for CUDA 10.2, and cuda101 for CUDA 10.1. 

```sh
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-10.1/
```






