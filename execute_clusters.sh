#!/bin/bash

source /home/arccha/miniconda3/bin/activate fungus

prefix="2_3_500_100p"
clusters="5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100"
clusters="5"

for cluster in $clusters
do
        python3 experiments/train_model.py --prefix $prefix --clusters $cluster --bow
done
