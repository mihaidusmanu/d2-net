#!/usr/bin/env bash

if [[ $# != 2 ]]; then
    echo 'Usage: bash preprocess_megadepth.sh /path/to/megadepth /output/path'
    exit
fi

export dataset_path=$1
export output_path=$2

echo 0
ls $dataset_path/MegaDepth_v1_SfM | xargs -P 8 -I % sh -c 'echo %; python preprocess_scene.py --base_path $dataset_path --scene_id % --subscene_id 0 --output_path $output_path'
echo 1
ls $dataset_path/MegaDepth_v1_SfM | xargs -P 8 -I % sh -c 'echo %; python preprocess_scene.py --base_path $dataset_path --scene_id % --subscene_id 1 --output_path $output_path'
echo 2
ls $dataset_path/MegaDepth_v1_SfM | xargs -P 8 -I % sh -c 'echo %; python preprocess_scene.py --base_path $dataset_path --scene_id % --subscene_id 2 --output_path $output_path'
echo 3
ls $dataset_path/MegaDepth_v1_SfM | xargs -P 8 -I % sh -c 'echo %; python preprocess_scene.py --base_path $dataset_path --scene_id % --subscene_id 3 --output_path $output_path'
