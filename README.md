# fungus

## Database

Database can be acquired via: https://drive.google.com/file/d/1-CjXE-HO3xz36x_XNJVDTRV89xDnjLlu/view?usp=sharing 

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

Then issue

```
python experiments/extract_features.py PATH_TO_IMAGES PATH_TO_MASKS --prefix PREFIX --size PATCH_SIZE
python experiments/extract_features.py PATH_TO_IMAGES PATH_TO_MASKS --prefix PREFIX --size PATCH_SIZE --test
python experiments/hyperparameters.py --prefix PREFIX
python experiments/confusion_matrices.py --prefix PREFIX
```

Parameters meaning can be checked in each script's documentation.

To run the experiments with neural networks, use branch `feature/neural_networks`.

To obtain the analysis of SVM classifier use notebook `SVM analysis.ipynb`.
