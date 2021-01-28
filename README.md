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
|`make doc`| create documentation with doxygen
|`make clean`| clean objects

## Run
Once builded the source you can just launch the executables passing the mandatory arguments N and K:
- `./mlp_omp N K`
- `./mlp_cuda N K`

## Profile
In order to profile the programs on the local machine just run:

`profile/local_profile.sh`

To profile the OMP implementation on a remote machine with ssh server enabled just run:

`profile/gcp_profile.sh USER HOST CERTIFICATE`

To profile the CUDA implementation on a Google Colab instance with GPU:

- open the `profile/colab_norebook.ipynb` on a Colab instance
- set up a machine with public ip and ssh server enabled that accept a certificate as credential
- upload on the colab instance BOTH public key and the certificate
- run the first two cells, setting the actual server ip
- on the local machine, just run `profile/colab_profile.sh SERVER_USER SERVER_HOST CERTIFICATE`
___