#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR


n_test=( 1000000 )

Order=( 3 4 )


for O in "${Order[@]}"
do
    for ndx in "${!n_test[@]}"
    do
        store_path=$SCRIPT_DIR"/Dim"$O
        store=$store_path
        if [ ! -d $store ]; then
            mkdir -p $store
        fi
        python data_gen_test.py --data-store-path=$store_path"/Allen_Cahen_Test"${n_test[$ndx]}".npz"D --x-range="0.0,1.0" --n-order=$O
    done
done