# Architectures and platforms for AI -- Module 2
## OpenMP and CUDA implementations of a MLP (2020/2021)
### by Daniele Verì
___
## Requirements
- `gcc` and `make` (both found in build-essential)
- `nvcc`
___
## Intructions
|command|description|
|-|-|
|`make`| build all
|`make openmp`| build openMP implementation
|`make openmp_dbg`| build openMP debug
|`make cuda`| build CUDA implementation
|`make cuda_dbg`| build CUDA debug
|`make cuda_legacy`| build CUDA for arch `compute_20` (cuda version < 9)
|`make clean`| clean objects
___
## TODO
- performance considerations in logging