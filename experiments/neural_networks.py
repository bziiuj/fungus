#!/usr/bin/env python
"""
Use neural networks to train a classifier of fungi microscopix images.
"""
import argparse
import copy
import functools
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))  # isort:skip
sys.path.append('/mnt2/fungus/')

import numpy as np
import torch
import tqdm
from sklearn.externals import joblib
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn.functional import interpolate
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torchvision import transforms


from dataset import FungusDataset
from dataset.normalization import normalize_image
from util.augmentation import NumpyToTensor
from pipeline.models import nn_models


@functools.lru_cache(maxsize=8)
def read_means_and_standard_deviations(means_path, stds_path):
    return np.load(means_path), np.load(stds_path)


class Upsample:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        return interpolate(sample.unsqueeze(0), size=self.size, mode='bicubic')[0]


def initialize_data_loader(args, shuffle):
    t = []

    if args.model == 'inceptionv3':
        t.insert(0, Upsample(299))
    if len(t) > 0:
        transform = transforms.Compose(t)
        tv = t
        # tv.insert(0, NumpyToTensor())
        tv.insert(1, normalize_image)
        transform_valid = transforms.Compose(tv)
    else:
        transform = None
        t.insert(0, NumpyToTensor())
        t.insert(1, normalize_image)
        transform_valid = transforms.Compose(t)

    train_dataset = FungusDataset(
        transform=transform,
        pngs_dir=args.pngs_path,
        masks_dir=args.masks_path,
        random_crop_size=args.size,
        number_of_bg_slices_per_image=4,
        number_of_fg_slices_per_image=16,
        train=True)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(torch.initial_seed()))

    test_dataset = FungusDataset(
        transform=transform_valid,
        pngs_dir=args.pngs_path,
        masks_dir=args.masks_path,
        random_crop_size=args.size,
        number_of_bg_slices_per_image=2,
        number_of_fg_slices_per_image=8,
        train=False,
        use_augmentation=False,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=shuffle,
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
        print(inputs.shape)
        loss, outputs = calculate_inception_loss(model, inputs, labels, criterion)
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
    parser.add_argument('--size', default=122, type=int, help='random crop radius')
    parser.add_argument('--epoch_no', default=100, type=int, help='no of epochs')
    args = parser.parse_args()
    filename_prefix = '{}/{}/{}'.format(args.results_dir, args.prefix, 'train')
    sm = SummaryWriter(log_dir=os.path.join('/mnt/tensorboard/', args.model))

    if args.test:
        shuffle = True
    else:
        shuffle = False

    dl_train, dl_test = initialize_data_loader(args, shuffle)

    model, is_inception = nn_models[args.model]
    optim = Adam(model.parameters(), lr=0.1)
    lr = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2,
                           verbose=False, threshold=0.0001, threshold_mode='rel',
                           cooldown=0, min_lr=0, eps=1e-04)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()

    if args.test:
        model_path = '{}{}_{}_best_model.pkl'.format(args.results_dir, filename_prefix.split('/')[-1], args.model)
        pred_label_path = '{}{}_{}_pred_label.pkl'.format(args.results_dir, filename_prefix.split('/')[-1], args.model)
        model.load_state_dict(joblib.load(model_path))
        model.eval()

        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        pred_label = []
        for _, d in tqdm.tqdm(enumerate(dl_test)):
            inputs = d[0]
            labels = d[1]
            if torch.cuda.is_available():
                inputs = inputs.cuda().float()
                labels = labels.cuda()

            with torch.set_grad_enabled(False):

                loss, outputs, preds = iterate_through_network(criterion, inputs, False, labels, model)

            # statistics
            epoch_val_loss += loss.item() * inputs.size(0)
            epoch_val_acc += torch.sum(preds == labels.data) / len(dl_test)
            pred_label.append([preds.detach().cpu().numpy(), labels.data.detach().cpu().numpy(), d[2]])
        np.savez_compressed(pred_label_path, pred_label=pred_label)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', epoch_val_loss, epoch_val_acc))
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
            print('Epoch: ', epoch_no)
            for _, d in tqdm.tqdm(enumerate(dl_train)):
                inputs = d[0].float()
                labels = d[1]
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
            model.eval()
            for _, d in tqdm.tqdm(enumerate(dl_test)):

                inputs = d[0].float()
                labels = d[1]
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

            lr.step(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', epoch_val_loss, epoch_val_acc))
            if epoch_val_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                joblib.dump(best_model_wts, '{}/{}_{}_best_model.pkl'.format(
                    args.results_dir, filename_prefix.split('/')[-1], args.model
                ))
            sm.add_scalar('loss_train', epoch_loss, epoch_no)
            sm.add_scalar('acc_train', epoch_acc, epoch_no)
            sm.add_scalar('loss_test', epoch_val_loss, epoch_no)
            sm.add_scalar('acc_test', epoch_val_acc, epoch_no)


if __name__ == '__main__':
    main()
