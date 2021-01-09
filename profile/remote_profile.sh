#!/bin/bash 

SSH_CERT=~/.ssh/gcloud.pem
LOCAL_BIND=127.0.0.6
USER_at_HOST=daniele.veri.96@35.238.172.110

MLP_OMP_EXE=/home/dan/Desktop/apai/mlp_omp
MLP_CUDA_EXE=/home/dan/Desktop/apai/mlp_cuda
PERF=/home/dan/Desktop/linux/tools/perf/perf

# setup local ssh tunnel
ssh -vv -o StrictHostKeyChecking=no -i $SSH_CERT \
-L $LOCAL_BIND:2022:localhost:3022 \
-L $LOCAL_BIND:49152:localhost:39152 \
-N $USER_at_HOST & > /dev/null

TUNNEL_PID=$(jobs -p)
sleep 5

# copy files host to remote 
scp -P 2022 -o StrictHostKeyChecking=no -i $SSH_CERT \
profile.sh \
$MLP_OMP_EXE \
$MLP_CUDA_EXE \
root@$LOCAL_BIND:~  # copy $PERF too?

# remote execution
ssh -i $SSH_CERT -o StrictHostKeyChecking=no root@$LOCAL_BIND -p 2022 \
MLP_OMP_EXE=./mlp_omp \
MLP_CUDA_EXE=./mlp_cuda \
N=100000 \
K=10 \
PERF=./perf \
NVPROF=/usr/local/cuda/bin/nvprof \
./profile.sh

# copy output logs back to localhost
scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no root@$LOCAL_BIND:nvprof_local.txt out/nvprof_remote.txt 
scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no root@$LOCAL_BIND:perf_local.txt out/perf_remote.txt 
scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no root@$LOCAL_BIND:cpu_local.txt out/cpu_remote.txt 
scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no root@$LOCAL_BIND:gpu_local.txt out/gpu_remote.txt 
scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no root@$LOCAL_BIND:strong_scaling_local.txt out/strong_scaling_remote.txt 
scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no root@$LOCAL_BIND:weak_scaling_local.txt out/weak_scaling_remote.txt 

kill $TUNNEL_PID
echo "SSH tunnel shutdown" $TUNNEL_PID
echo "-- Remote profiling completed --"