#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR

declare -i count=2

n_train_col=( 48000 )
n_train_bound=( 9600 )

n_train_batch_col=( 100000 )
n_train_batch_bound=( 100000 )

ks=( 8 )

jitt=( 21.587 )

ls=( -2.7899 )
rank=( 19 )
ampl=( 1.0 )

trainGrid=( 71 )

ALPHA=( 1505.0 )
BETA=( 90000000000.0 )

dataN=( 5 )

store_path=$SCRIPT_DIR"/result/Data_AllenC6D_CP_Efficiency"
model_path=$SCRIPT_DIR"/training_process_store_pth"

dataset_path=$SCRIPT_DIR"/dataset"

for nxdx in "${!trainGrid[@]}"
do
    store=$store_path"/Result_C"${n_train_col[$nxdx]}"_B"$((((${n_train_bound[$nxdx]} + 6 * 2 - 1)) / ((6 * 2)) * ((6 * 2))))
    if [ ! -d $store ]; then
        mkdir -p $store
    fi
    data=$store_path"/Data_C"${n_train_col[$nxdx]}"_B"$((((${n_train_bound[$nxdx]} + 6 * 2 - 1)) / ((6 * 2)) * ((6 * 2))))
    if [ ! -d $data ]; then
        mkdir -p $data
    fi
    best=$model_path"/Model_C"${n_train_col[$nxdx]}"_B"$((((${n_train_bound[$nxdx]} + 6 * 2 - 1)) / ((6 * 2)) * ((6 * 2))))"/best_model_pth"
    if [ ! -d $best ]; then
        mkdir -p $best
    fi
    training=$model_path"/Model_C"${n_train_col[$nxdx]}"_B"$((((${n_train_bound[$nxdx]} + 6 * 2 - 1)) / ((6 * 2)) * ((6 * 2))))"/training_store_pth"
    if [ ! -d $training ]; then
        mkdir -p $training
    fi
    for ddx in "${!dataN[@]}"
    do
        python GP_S.py --lr=0.01 --lr-gap=1000 --lr-decay=0.8 --epochs=1000000 --early-stop=20000 --log-store-path=$store"/config"$count \
         --kernel-s ${ks[$nxdx]} ${ks[$nxdx]} ${ks[$nxdx]} ${ks[$nxdx]} ${ks[$nxdx]} ${ks[$nxdx]} --jitter ${jitt[$nxdx]} ${jitt[$nxdx]} ${jitt[$nxdx]} ${jitt[$nxdx]} ${jitt[$nxdx]} ${jitt[$nxdx]} \
         --train-batch-boundary=${n_train_batch_bound[$nxdx]} --lsx ${ls[$nxdx]} ${ls[$nxdx]} ${ls[$nxdx]} ${ls[$nxdx]} ${ls[$nxdx]} ${ls[$nxdx]} \
         --train-batch-collocation=${n_train_batch_col[$ndx]} --rank ${rank[$nxdx]} ${rank[$nxdx]} ${rank[$nxdx]} ${rank[$nxdx]} ${rank[$nxdx]} ${rank[$nxdx]} \
         --x-range="0.0,1.0" --n-xtrain ${trainGrid[$nxdx]} ${trainGrid[$nxdx]} ${trainGrid[$nxdx]} ${trainGrid[$nxdx]} ${trainGrid[$nxdx]} ${trainGrid[$nxdx]} --test-batch=1000000 --method=2 \
         --n-train-collocation=${n_train_col[$nxdx]} --n-train-boundary=${n_train_bound[$nxdx]} --log-interval=1000 --alpha=${ALPHA[$nxdx]} --beta=${BETA[$nxdx]} --n-train-batch=48000 \
         --n-order=6 --seed=0 --dataset-load-path=$dataset_path"/Dim6_"${dataN[$ddx]}"/Allen_CahenC"${n_train_col[$nxdx]}"_B"$((((${n_train_bound[$nxdx]} + 6 * 2 - 1)) / ((6 * 2)) * ((6 * 2))))".npz" \
         --amplitude ${ampl[$nxdx]} ${ampl[$nxdx]} ${ampl[$nxdx]} ${ampl[$nxdx]} ${ampl[$nxdx]} ${ampl[$nxdx]} --best-model-dir=$best"/best_model"$count".pth" \
         --test-dataset-load-path=$dataset_path"/Allen_Cahen_Test1000000.npz" --checkpoint-dir=$training"/checkpoint"$count".pth" --process-store-path=$data"/process"$count".npz" --regulator
        count=$(( count + 1 ))
    done
    count=0
done
