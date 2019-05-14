#!/usr/bin/env python
"""
Read feature matrix and labels from .npy files and classify them. In train
mode use train dataset, fit GMM and then fit SVC, in test mode load best
model obtained from `hyperparameters.py` and perform prediction on test
dataset.
"""
import argparse
import copy
import functools

import numpy as np
import torch
from sklearn.externals import joblib
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torchvision import transforms


from dataset import FungusDataset
from pipeline.models import nn_models
from util.rotation_by90 import RotationBy90

import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip


@functools.lru_cache(maxsize=8)
def read_means_and_standard_deviations(means_path, stds_path):
    return np.load(means_path), np.load(stds_path)


def initialize_data_loader(args):
    means, stds = read_means_and_standard_deviations(
        'tmp/means.npy', 'tmp/stds.npy')
    transform = transforms.Compose([
        RotationBy90(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    train_dataset = FungusDataset(
        transform=transform,
        pngs_dir=args.pngs_path,
        masks_dir=args.masks_path,
        random_crop_size=args.size,
        number_of_bg_slices_per_image=16,
        number_of_fg_slices_per_image=4,
        train=True)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(torch.initial_seed()))

    test_dataset = FungusDataset(
        transform=transform,
        pngs_dir=args.pngs_path,
        masks_dir=args.masks_path,
        random_crop_size=args.size,
        number_of_bg_slices_per_image=16,
        number_of_fg_slices_per_image=4,
        train=False)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(torch.initial_seed()))
    
    return train_loader, test_loader


def calculate_inception_loss(model, inputs, labels, criterion):
    outputs, aux_outputs = model(inputs)
    loss1 = criterion(outputs, labels)
    loss2 = criterion(aux_outputs, labels)
    return loss1 + 0.4 * loss2, outputs


def iterate_through_network(criterion, inputs, is_inception, labels, model):
    if is_inception:
        outputs, loss = calculate_inception_loss(model, inputs, labels, criterion)
    else:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    _, preds = torch.max(outputs, 1)
    return loss, outputs, preds


def main():
    SEED = 9001
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'results_dir', help='absolute path to results directory')
    parser.add_argument('--test', default=False,
                        action='store_true', help='enable test mode')
    parser.add_argument('--prefix', default='', help='result filenames prefix')
    parser.add_argument('--model', default='alexnet', help='nn model')
    parser.add_argument('--pngs_path', help='absolute path to directory with pngs')
    parser.add_argument('--masks_path', help='absolute path to directory with masks')
    parser.add_argument('--size', default=125, type=int, help='random crop radius')
    parser.add_argument('--epoch_no', default=50, type=int, help='no of epochs')
    args = parser.parse_args()
    filename_prefix = '{}/{}/{}'.format(args.results_dir, args.prefix, 'test' if args.test else 'train')
    sm = SummaryWriter(log_dir=os.path.join('/tensorboard/', 'args.model'))

    dl_train, dl_test = initialize_data_loader(args)

    model, is_inception = nn_models[args.model]
    optim = Adam(model.parameters(), lr=0.1)
    lr = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5,
                           verbose=False, threshold=0.0001, threshold_mode='rel',
                           cooldown=0, min_lr=0, eps=1e-04)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()

    if args.test:
        pass
    else:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch_no in range(args.epoch_no):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            running_val_loss = 0.0
            running_val_corrects = 0

            # Iterate over data.
            for inputs, labels in dl_train:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                    with torch.set_grad_enabled(True):

                        loss, outputs, preds = iterate_through_network(criterion, inputs, is_inception, labels, model)
                        loss.backward()
                        optim.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dl_train.dataset)
            epoch_acc = running_corrects.double() / len(dl_train.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

            for inputs, labels in dl_test:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                    with torch.set_grad_enabled(False):

                        loss, outputs, preds = iterate_through_network(criterion, inputs, is_inception, labels, model)

                    # statistics
                    running_val_loss += loss.item() * inputs.size(0)
                    running_val_corrects += torch.sum(preds == labels.data)

            epoch_val_loss = running_val_loss / len(dl_test.dataset)
            epoch_val_acc = running_val_corrects.double() / len(dl_test.dataset)

            lr.step(epoch_val_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', epoch_val_loss, epoch_val_acc))
            if epoch_val_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            sm.add_scalar('loss_train', epoch_loss, epoch_no)
            sm.add_scalar('acc_train', epoch_acc, epoch_no)
            sm.add_scalar('loss_test', epoch_val_loss, epoch_no)
            sm.add_scalar('acc_test', epoch_val_acc, epoch_no)
        joblib.dump(best_model_wts, '{}/{}_{}_best_model.pkl'.format(args.results_dir, filename_prefix, args.model))


if __name__ == '__main__':
    main()