#!/bin/bash

echo "- Detecting devices info ..."
lscpu > cpu_local.txt
cd /usr/local/cuda/samples/1_Utilities/deviceQuery && make > /dev/null && cd - > /dev/null
/usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery > gpu_local.txt

# openmp statistics
echo "- Running OPENMP perf ..."
$PERF stat -B -o perf_local.txt -e \
task-clock,cycles,branches,branch-misses,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
L1-dcache-stores,L1-dcache-store-misses \
$MLP_OMP_EXE $N $K > /dev/null

# cuda statistics (local GPU: MX130)
echo "- Running CUDA nvprof ..."
$NVPROF --events all --metrics all --log-file nvprof_local.txt \
$MLP_CUDA_EXE $N $K > /dev/null

# strong scaling
echo "- Performing OPENMP strong scaling test ..."
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores
echo "average time elapsed for N=$N K=$K" > strong_scaling_local.txt
for p in `seq $CORES`; do
    sum=0
    count=0
    for rep in `seq 5`; do
        EXEC_TIME="$(OMP_NUM_THREADS=$p $MLP_OMP_EXE $N $K | grep -Po '(?<=(P1 time elapsed = )).*(?= s)')"
        sum=$(echo "$sum + $EXEC_TIME" | bc)
        count=$(echo "$count + 1" | bc)
    done
    avg=$(echo "scale=3; $sum / $count" | bc)
    echo "$p cores: $avg" >> strong_scaling_local.txt
done

# weak scaling
echo "- Performing OPENMP weak scaling test ..."
N0=$N # base problem size
echo "average time elapsed for K = $K" > weak_scaling_local.txt
for p in `seq $CORES`; do
    PROB_SIZE=`echo "e(l($N0 * $N0 * $p)/2)" | bc -l -q`
    PROB_SIZE=`echo "$PROB_SIZE / 1" | bc`
    sum=0
    count=0
    for rep in `seq 5`; do
        EXEC_TIME="$(OMP_NUM_THREADS=$p $MLP_OMP_EXE $PROB_SIZE $K | grep -Po '(?<=(P1 time elapsed = )).*(?= s)')"
        sum=$(echo "$sum + $EXEC_TIME" | bc)
        count=$(echo "$count + 1" | bc)
    done
    avg=$(echo "scale=3; $sum / $count" | bc)
    echo "$p cores, $PROB_SIZE N: $avg" >> weak_scaling_local.txt
done
