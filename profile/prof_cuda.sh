#!/bin/bash

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

echo "- Detecting GPU info ..."
cd /usr/local/cuda/samples/1_Utilities/deviceQuery && make > /dev/null && cd - > /dev/null
/usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery > $SCRIPTPATH/gpu_local.txt

# cuda profiling
#echo "- Running CUDA nvprof ..."
#$NVPROF --events all --metrics all --log-file $SCRIPTPATH/nvprof_local.txt \
#$MLP_CUDA_EXE $N $K > /dev/null

# throughput
echo "- Performing throughput test ..."
Ni=1000 # base problem size
K=100
echo "throughput for K = $K" > $SCRIPTPATH/throughput_local.txt
for rep in `seq 15`; do
    Ni=`echo "$Ni*2" | bc -l -q`
    EXEC_TIME="$($MLP_CUDA_EXE $Ni $K | grep -Po '(?<=(P1 time elapsed = )).*(?= s)')"
    PROB_SIZE=`echo "($Ni-2*($K+1))*$K" | bc -l -q` # assuming R=5
    THROUGHPUT=$(echo "scale=3; $PROB_SIZE/$EXEC_TIME" | bc)
    echo "$Ni N: $THROUGHPUT" >> $SCRIPTPATH/throughput_local.txt
done
