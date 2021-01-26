# Architectures and platforms for AI -- Module 2
## OpenMP and CUDA implementations of a locally connected MLP (2020/2021)
### by Daniele Verì
___
## Requirements
- `gcc` and `make` (both found in build-essential)
- `nvcc`
___
## Build
|command|description|
|-|-|
|`make`| build all
|`make openmp`| build openMP implementation
|`make openmp_dbg`| build openMP debug
|`make cuda`| build CUDA implementation
|`make cuda_dbg`| build CUDA debug
|`make cuda_legacy`| build CUDA for arch `compute_20` (cuda version < 9)
|`make clean`| clean objects

## Run
Once builded the source you can just launch the executables passing the mandatory arguments N and K:
- `./mlp_omp N K`
- `./mlp_cuda N K`

## Profile
In order to profile the programs on the local machine just run:

`profile/local_profile`
___
## TODO
- comments
- omp tiling
- cuda cache/shared
