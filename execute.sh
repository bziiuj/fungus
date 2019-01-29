#!/bin/bash

source /home/arccha/miniconda3/bin/activate fungus

png100p_dir="/home/arccha/fungus_data_png/pngs/"
png50p_dir="/home/arccha/fungus_data_png/pngs_50p/"
mask13_dir="/home/arccha/fungus_data_png/masks_1_3/"
mask12_dir="/home/arccha/fungus_data_png/masks_1_2/"
mask23_dir="/home/arccha/fungus_data_png/masks_2_3/"
mask1350p_dir="/home/arccha/fungus_data_png/masks_1_3_50p/"
mask1250p_dir="/home/arccha/fungus_data_png/masks_1_2_50p/"
mask2350p_dir="/home/arccha/fungus_data_png/masks_2_3_50p/"
masks="1_3 1_2 2_3"
sizes="125 250 500"
scales="0.5 1.0"
results_dir="/home/arccha/fungus_results/"
bow_type="fv"

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
                                png_dir=$png100p_dir
			else
				prefix=${prefix}_50p
                                png_dir=$png50p_dir
			fi
			if [ $mask = "1_3" ]
			then
                            if [ $scale = "1.0" ]
                            then
                              mask_path=$mask13_dir
                            else
                              mask_path=$mask1350p_dir
                            fi
			elif [ $mask = "1_2" ]
			then
			    if [ $scale = "1.0" ]
                            then
                              mask_path=$mask12_dir
                            else
                              mask_path=$mask1250p_dir
                            fi
			else
			    if [ $scale = "1.0" ]
                            then
                              mask_path=$mask23_dir
                            else
                              mask_path=$mask2350p_dir
                            fi
			fi
			echo "Computing ${prefix}..."
                        mkdir -p "${results_dir}/${prefix}"
			python3 experiments/extract_features.py $png_dir $mask_path $results_dir --prefix $prefix --size $size
			python3 experiments/extract_features.py $png_dir $mask_path $results_dir --prefix $prefix --size $size --test
			python3 experiments/hyperparameters.py $results_dir --prefix $prefix
			python3 experiments/confusion_matrices.py $results_dir --prefix $prefix
		done
	done
done
