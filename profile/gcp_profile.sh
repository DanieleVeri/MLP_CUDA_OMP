#!/bin/bash 

USER$1
HOST=$2
SSH_CERT=$3
LOCAL_BIND=127.0.0.6

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

# setup local ssh tunnel
echo "- Setting up tunnel ssh ..."
ssh -o StrictHostKeyChecking=no -i $SSH_CERT \
-L $LOCAL_BIND:2022:localhost:22 \
-N $USER@$HOST & > /dev/null

TUNNEL_PID=$(jobs -p)
sleep 5

# copy files host to remote 
echo "- Deploying files in remote ..."
scp -P 2022 -o StrictHostKeyChecking=no -i $SSH_CERT \
$SCRIPTPATH/prof_omp.sh \
$SCRIPTPATH/../mlp_omp \
$USER@$LOCAL_BIND:~

# remote execution
ssh -i $SSH_CERT -o StrictHostKeyChecking=no $USER@$LOCAL_BIND -p 2022 \
MLP_OMP_EXE=./mlp_omp \
N=1000000 \
K=100 \
PERF=./perf \
./prof_omp.sh

# copy output logs back to localhost
echo "- Copyings logs from remote ..."
scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no \
$USER@$LOCAL_BIND:perf_local.txt $SCRIPTPATH/out/perf_remote.txt 

scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no \
$USER@$LOCAL_BIND:cpu_local.txt $SCRIPTPATH/out/cpu_remote.txt 

scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no \
$USER@$LOCAL_BIND:strong_scaling_local.txt $SCRIPTPATH/out/strong_scaling_remote.txt 

scp -P 2022 -i $SSH_CERT -o StrictHostKeyChecking=no \
$USER@$LOCAL_BIND:weak_scaling_local.txt $SCRIPTPATH/out/weak_scaling_remote.txt 

echo "- SSH tunnel shutdown ($TUNNEL_PID)"
kill $TUNNEL_PID

echo "-- Remote profiling completed --"