#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR

declare -i count=0


n_train_col=( 96000 48000 32000 16000 )
n_train_bound=( 19200 9600 6400 3200 )


# n_train_col=( 24000 )
# n_train_bound=( 4800 )

Order=( 6 )

for i in {0..6}
do
    for O in "${Order[@]}"
    do
        for ndx in "${!n_train_col[@]}"
        do
        # for ((ndx=${#n_train_col[@]}-1; ndx>=0; ndx--))
        # do
            store_path=$SCRIPT_DIR"/Dim"$O"_"$count
            store=$store_path
            if [ ! -d $store ]; then
                mkdir -p $store
            fi
            python data_gen.py --data-store-path=$store_path"/Allen_CahenC"${n_train_col[$ndx]}"_B"$((((${n_train_bound[$ndx]} + $O * 2 - 1)) / (($O * 2)) * (($O * 2))))".npz" \
             --x-range="0.0,1.0" --n-train-collocation=${n_train_col[$ndx]} --n-train-boundary=${n_train_bound[$ndx]} --n-order=$O --seed=$RANDOM
        done
    done
    count=$(( count + 1 ))
done