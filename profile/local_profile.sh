#!/bin/bash

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
mkdir -p $SCRIPTPATH/out

MLP_OMP_EXE=$SCRIPTPATH/../mlp_omp \
N=100000 \
K=100 \
PERF=perf \
$SCRIPTPATH/prof_omp.sh

MLP_CUDA_EXE=$SCRIPTPATH/../mlp_cuda \
N=100000 \
K=100 \
NVPROF=/usr/local/cuda/bin/nvprof \
$SCRIPTPATH/prof_cuda.sh

mv $SCRIPTPATH/*.txt $SCRIPTPATH/out/

echo "-- Local profiling completed --"