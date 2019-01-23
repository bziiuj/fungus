#!/bin/bash

source /home/user/anaconda3/bin/activate fungus

png_dir="/media/data2/fungus/data/50p/pngs/"
mask_path="/media/data2/fungus/data/50p/masks_2_3/"
results_dir="/media/data2/fungus/results/"
masks="2_3"
size="250"
scale="0.5"

prefix=${mask}_${size}
if [ $scale = "1.0" ]
then
    prefix=${prefix}_100p
else
    prefix=${prefix}_50p
fi
echo "Computing ${prefix}..."
python3 dataset/normalization.py $png_dir
python3 experiments/extract_features.py $png_dir $mask_path $results_dir --prefix $prefix --size $size
python3 experiments/extract_features.py $png_dir $mask_path $results_dir --prefix $prefix --size $size --test
python3 experiments/predict_classification.py $results_dir --prefix $prefix
python3 experiments/predict_classification.py $results_dir --prefix $prefix --test
python3 experiments/confusion_matrices.py $results_dir --prefix $prefix
python3 experiments/visual_histograms.py $results_dir --prefix $prefix
