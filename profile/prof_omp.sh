#!/bin/bash

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

echo "- Detecting CPU info ..."
lscpu > $SCRIPTPATH/cpu_local.txt

# perf profiler statistics
echo "- Running OPENMP perf ..."
$PERF stat -B -o $SCRIPTPATH/perf_local.txt -e \
task-clock,cycles,branches,branch-misses,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
L1-dcache-stores,L1-dcache-store-misses \
$MLP_OMP_EXE $N $K > /dev/null

# Amdahl's law
echo "- Performing maximum speedup test ..."
sum=0
count=0
for rep in `seq 3`; do
    OUT="$(OMP_NUM_THREADS=1 $MLP_OMP_EXE $N $K)"
    NPT="$(echo $OUT | grep -Po '(?<=(Not parallelized time = )).*(?= s P)')"
    TT="$(echo $OUT | grep -Po '(?<=(Parallel time elapsed = )).*(?= s)')"
    sum=$(echo "scale=3; $sum + ($NPT / $TT)" | bc)
    count=$(echo "$count + 1" | bc)
done
ALPHA=$(echo "scale=3; $sum / $count" | bc)
ASINT=$(echo "scale=3; 1/$ALPHA" | bc)
echo "Asintotic speedup according Amdahl law: $ASINT" > $SCRIPTPATH/amdahl_local.txt
for cores in `seq 16`; do
    SPEEDUP=$(echo "scale=3; 1 / ($ALPHA + (1-$ALPHA)/$cores)" | bc)
    echo "cores: $cores -> speedup: $SPEEDUP" >> $SCRIPTPATH/amdahl_local.txt
done

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