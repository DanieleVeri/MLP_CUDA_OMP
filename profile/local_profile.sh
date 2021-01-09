#!/bin/bash

MLP_OMP_EXE=/home/dan/Desktop/apai/mlp_omp \
MLP_CUDA_EXE=/home/dan/Desktop/apai/mlp_cuda \
N=100000 \
K=10 \
PERF=/home/dan/Desktop/linux/tools/perf/perf \
NVPROF=/usr/local/cuda/bin/nvprof \
./profile.sh

mv *.txt out/

echo "-- Local profiling completed --"