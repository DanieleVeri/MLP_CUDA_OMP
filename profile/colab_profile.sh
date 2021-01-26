#!/bin/bash 

USER=$1
HOST=$2
SSH_CERT=$3
LOCAL_BIND=127.0.0.6

echo "@@@@@@ WARNING: server fingerprint verification ignored @@@@@@"
echo "@@@@@@ DO NOT USE this script for SENSITIVE data!       @@@@@@"

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

# setup local ssh tunnel
echo "- Setting up tunnel ssh ..."
ssh -o StrictHostKeyChecking=no -i $SSH_CERT \
-L $LOCAL_BIND:2022:localhost:3022 \
-L $LOCAL_BIND:49152:localhost:39152 \
-N $USER@$HOST & > /dev/null

TUNNEL_PID=$(jobs -p)
sleep 5

# copy files host to remote 
echo "- Deploying files in remote ..."
scp -P 2022 -o StrictHostKeyChecking=no -i $SSH_CERT \
$SCRIPTPATH/prof_cuda.sh \
$SCRIPTPATH/../mlp_cuda \
root@$LOCAL_BIND:~

# remote execution
ssh -i $SSH_CERT -o StrictHostKeyChecking=no root@$LOCAL_BIND -p 2022 \
MLP_CUDA_EXE=./mlp_cuda \
N=1000000 \
K=100 \
NVPROF=/usr/local/cuda/bin/nvprof \
./prof_cuda.sh

# copy output logs back to localhost
echo "- Copyings logs from remote ..."
scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no \
root@$LOCAL_BIND:nvprof_local.txt $SCRIPTPATH/out/nvprof_remote.txt 

scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no \
root@$LOCAL_BIND:gpu_local.txt $SCRIPTPATH/out/gpu_remote.txt 

scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no \
root@$LOCAL_BIND:throughput_local.txt $SCRIPTPATH/out/throughput_remote.txt 

echo "- SSH tunnel shutdown ($TUNNEL_PID)"
kill $TUNNEL_PID

echo "-- Remote profiling completed --"