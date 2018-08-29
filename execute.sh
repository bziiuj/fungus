#!/bin/bash

source /home/arccha/miniconda3/bin/activate fungus 

png_dir="/home/arccha/fungus_data_png/pngs/"
mask13_dir="/home/arccha/fungus_data_png/masks_1_3/"
mask12_dir="/home/arccha/fungus_data_png/masks_1_2/"
mask23_dir="/home/arccha/fungus_data_png/masks_2_3/"
masks="1_3 1_2 2_3"
sizes="125 250 500"
scales="1.0 0.5"

for mask in $masks
do
	for size in $sizes
	do
		for scale in $scales
		do
			prefix=${mask}_${size}
			if [ $scale = "1.0" ]
			then
				prefix=${prefix}_100p
			else
				prefix=${prefix}_50p
			fi
			if [ $mask = "1_3" ]
			then
				mask_path=$mask13_dir
			elif [ $mask = "1_2" ]
			then
				mask_path=$mask12_dir
			else
				mask_path=$mask23_dir
			fi
			echo "Computing ${prefix}..."
			python3 experiments/extract_features.py $png_dir $mask_path --prefix $prefix --size $size --scale $scale
			python3 experiments/extract_features.py $png_dir $mask_path --prefix $prefix --size $size --scale $scale --test
			python3 experiments/hyperparameters.py --prefix $prefix
			python3 experiments/confusion_matrices.py --prefix $prefix
		done	
	done	
done	
