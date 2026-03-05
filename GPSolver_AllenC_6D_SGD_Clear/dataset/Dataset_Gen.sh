#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR


n_train_col=( 1000 2000 4000 8000 16000 32000 )
n_train_bound=( 200 400 800 1600 3200 6400 )

# n_train_col=( 24000 )
# n_train_bound=( 4800 )

Order=( 3 4 )

for O in "${Order[@]}"
do
    for ndx in "${!n_train_col[@]}"
    do
        store_path=$SCRIPT_DIR"/Dim"$O
        store=$store_path
        if [ ! -d $store ]; then
            mkdir -p $store
        fi
        python data_gen.py --data-store-path=$store_path"/Allen_CahenC"${n_train_col[$ndx]}"_B"$((((${n_train_bound[$ndx]} + $O * 2 - 1)) / (($O * 2)) * (($O * 2))))".npz" \
         --x-range="0.0,1.0" --n-train-collocation=${n_train_col[$ndx]} --n-train-boundary=${n_train_bound[$ndx]} --n-order=$O
    done
done