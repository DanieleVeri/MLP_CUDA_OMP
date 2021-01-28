#!/bin/bash

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

echo "- Detecting CPU info ..."
lscpu > $SCRIPTPATH/cpu_local.txt

# openmp statistics
echo "- Running OPENMP perf ..."
$PERF stat -B -o $SCRIPTPATH/perf_local.txt -e \
task-clock,cycles,branches,branch-misses,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
L1-dcache-stores,L1-dcache-store-misses \
$MLP_OMP_EXE $N $K > /dev/null

# strong scaling
echo "- Performing OPENMP strong scaling test ..."
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores
echo "average time elapsed for N=$N K=$K" > $SCRIPTPATH/strong_scaling_local.txt
for p in `seq $CORES`; do
    sum=0
    count=0
    for rep in `seq 5`; do
        EXEC_TIME="$(OMP_NUM_THREADS=$p $MLP_OMP_EXE $N $K | grep -Po '(?<=(Parallel time elapsed = )).*(?= s)')"
        sum=$(echo "$sum + $EXEC_TIME" | bc)
        count=$(echo "$count + 1" | bc)
    done
    avg=$(echo "scale=3; $sum / $count" | bc)
    echo "$p cores: $avg" >> $SCRIPTPATH/strong_scaling_local.txt
done

# weak scaling
echo "- Performing OPENMP weak scaling test ..."
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores
N0=`echo "($N-2*($K+1))*5*$K" | bc -l -q` # base problem size (assuming R=5)
echo "average time elapsed for K = $K" > $SCRIPTPATH/weak_scaling_local.txt
for p in `seq $CORES`; do
    PROB_SIZE=`echo "($N0*$p/5/$K)+2*($K+1)" | bc -l -q`
    PROB_SIZE=`echo "$PROB_SIZE / 1" | bc`
    sum=0
    count=0
    for rep in `seq 5`; do
        EXEC_TIME="$(OMP_NUM_THREADS=$p $MLP_OMP_EXE $PROB_SIZE $K | grep -Po '(?<=(Parallel time elapsed = )).*(?= s)')"
        sum=$(echo "$sum + $EXEC_TIME" | bc)
        count=$(echo "$count + 1" | bc)
    done
    avg=$(echo "scale=3; $sum / $count" | bc)
    echo "$p cores, $PROB_SIZE N: $avg" >> $SCRIPTPATH/weak_scaling_local.txt
done