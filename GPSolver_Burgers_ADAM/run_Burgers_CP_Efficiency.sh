#!/bin/bash

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR


declare -i count=1

n_x1=( 700 )
n_x2=( 40 )

n_xind1=( 718 )
n_xind2=( 41 )

ks1=( 1 )
ks2=( 2 )
jitt1=( 13.98 )
jitt2=( 13.9 )
ls1=( -2.99 )
ls2=( -1.11 )

n_train_batch_col=( 28000 )
n_train_batch_bound=( 1500 )

rank=( 18 )


store_path=$SCRIPT_DIR"/result/Data_Burgers0001_CP_Efficiency"
# store=$store_path
# if [ ! -d $store ]; then
#     mkdir -p $store
# fi

for nxdx in "${!n_x1[@]}"
do
    store=$store_path"/Result_nu0001_"${n_x1[$nxdx]}"x"${n_x2[$nxdx]}
    if [ ! -d $store ]; then
        mkdir -p $store
    fi
    data=$store_path"/Data_nu0001_"${n_x1[$nxdx]}"x"${n_x2[$nxdx]}
    if [ ! -d $data ]; then
        mkdir -p $data
    fi
    python GP_S.py --lr=0.01 --lr-gap=1000 --lr-decay=0.8 --epochs=1000000 --early-stop=10000 --log-store-path=$store"/config"$count \
     --kernel-s ${ks1[$nxdx]} ${ks2[$nxdx]} --jitter ${jitt1[$nxdx]} ${jitt2[$nxdx]} --method=2 --log-lsx ${ls1[$nxdx]} ${ls2[$nxdx]} \
     --rank ${rank[$nxdx]} ${rank[$nxdx]} --n-xtrain ${n_x1[$nxdx]} ${n_x2[$nxdx]} --log-interval=10000 --alpha=1000.0 --beta=2000000.0 --n-order=2 \
     --seed=0 --n-train-collocation=$((${n_x1[$nxdx]} * ${n_x2[$nxdx]})) --process-store-path=$data"/process"$count".npz" --n-xind ${n_xind1[$nxdx]} ${n_xind2[$nxdx]} \
     --train-batch-collocation=${n_train_batch_col[$nxdx]} --train-batch-boundary=${n_train_batch_bound[$nxdx]} --regulator
    count=$(( count + 1 ))
done
