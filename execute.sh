#!/bin/bash

source /opt/anaconda/bin/activate fungus

python experiments/extract_features.py /mnt/data/fungus/preprocessed /mnt/data/fungus/masks_2_3 --prefix 2_3_250_50p --size 250 --prescale 0.5 --augment
python experiments/extract_features.py /mnt/data/fungus/preprocessed /mnt/data/fungus/masks_2_3 --prefix 2_3_250_50p --size 250 --prescale 0.5 --test --augment
python experiments/hyperparameters.py --prefix 2_3_250_50p --augment
python experiments/confusion_matrices.py --prefix 2_3_250_50p --augment
