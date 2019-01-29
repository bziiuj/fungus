#!/bin/bash


png_dir="/home/arccha/tmp/fungus/pngs_50p/"
mask_path="/home/arccha/tmp/fungus/masks_2_3_50p/"
results_dir="/home/arccha/tmp/fungus_results/"
bow_type="fv"
mask="2_3"
size="250"
scale="50p"

prefix=${bow_type}_${mask}_${size}_${scale}
mkdir -p ${results_dir}/${prefix}
echo "Computing ${prefix}..."
python3 dataset/normalization.py $png_dir
python3 experiments/extract_features.py $png_dir $mask_path $results_dir --prefix $prefix --size $size
python3 experiments/extract_features.py $png_dir $mask_path $results_dir --prefix $prefix --size $size --test
python3 experiments/predict_classification.py $results_dir --prefix $prefix --gmm-clusters-number 10 --kernel linear --C 100 --gamma 'auto'
python3 experiments/predict_classification.py $results_dir --prefix $prefix --test
python3 experiments/confusion_matrices.py $results_dir --prefix $prefix
python3 experiments/visual_histograms.py $results_dir --prefix $prefix
