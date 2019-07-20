# fungus

## Conda environment

Conda environment can be set up by issuing:

```
conda env create -f environment.yml
```

## Running experiments

Firstly, you need to calculate parameters for normalization. You can do that by invoking

```
python dataset/normalization.py PATH_TO_IMAGES
```

Results will be saved in `results/{means.npy,stds.npy}`.

Then tou can run experiments with training and saving validation results to directory `../results` with fine-tuned models. 
Available models are: `alexnet`, `densenet`, `inceptionv3`, `resnet18` and `resnet50`.

```
python experiments/neural_networks.py --prefix PREFIX
```

Parameters meaning can be checked in each script's documentation.

To gather the results use notebook `experiments/aggregate_info.ipynb`
