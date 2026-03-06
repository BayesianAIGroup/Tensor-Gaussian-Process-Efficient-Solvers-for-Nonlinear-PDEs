#!/bin/bash

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR

declare -i count=1

n_x=( 70 )

n_xind=( 265 )

n_train_col=( 4900 )
n_train_bound=( 70 )

ks1=( 5 )
ks2=( 5 )
jitt1=( 22.5 )
jitt2=( 22.5 )
ls1=( -3.5 )
ls2=( -3.5 )

rank=( 10 )

a=( 15.0 )


store_path=$SCRIPT_DIR"/result/Data_AllenC_CP_SGD_efficiency"
# store=$store_path
# if [ ! -d $store ]; then
#     mkdir -p $store
# fi

for nxdx in "${!n_x[@]}"
do
    store=$store_path"/Result_a"${a[$nxdx]}"_"${n_x[$nxdx]}"x"${n_x[$nxdx]}
    if [ ! -d $store ]; then
        mkdir -p $store
    fi
    data=$store_path"/Data_a"${a[$nxdx]}"_"${n_x[$nxdx]}"x"${n_x[$nxdx]}
    if [ ! -d $data ]; then
        mkdir -p $data
    fi
    python GP_S.py --lr=0.01 --lr-gap=1000 --lr-decay=0.8 --epochs=1000000 --early-stop=20000 --log-store-path=$store"/config"$count \
     --kernel-s ${ks1[$nxdx]} ${ks2[$nxdx]} --jitter ${jitt1[$nxdx]} ${jitt2[$nxdx]} --method=2 --log-lsx ${ls1[$nxdx]} ${ls2[$nxdx]} \
     --rank ${rank[$nxdx]} ${rank[$nxdx]} --x-range="0.0,1.0" --n-xtrain ${n_x[$nxdx]} ${n_x[$nxdx]} --n-xind ${n_xind[$nxdx]} ${n_xind[$nxdx]} --n-x1test=100 \
     --n-x2test=100 --n-train-collocation=$((${n_x[$nxdx]} * ${n_x[$nxdx]})) --n-testing=10000 --log-interval=10000 \
     --alpha=6500.0 --beta=20000000.0 --n-order=2 --seed=0 --process-store-path=$data"/process"$count".npz"
    count=$(( count + 1 ))
done
