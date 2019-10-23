#!/bin/bash

source /opt/anaconda/bin/activate fungus

python dataset/normalization.py /mnt/data/fungus/pngs_50p
python experiments/extract_features.py /mnt/data/fungus/pngs_50p /mnt/data/fungus/masks_2_3_50p --prefix 2_3_250_50p --size 250 --augment
python experiments/extract_features.py /mnt/data/fungus/pngs_50p /mnt/data/fungus/masks_2_3_50p --prefix 2_3_250_50p --size 250 --test --augment
python experiments/hyperparameters.py --prefix 2_3_250_50p --augment
python experiments/confusion_matrices.py --prefix 2_3_250_50p --augment
