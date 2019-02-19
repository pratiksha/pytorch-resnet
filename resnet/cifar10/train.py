import os
from datetime import datetime
from collections import OrderedDict

import click
import torch
import tqdm
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import datasets

from resnet import utils, file_utils, data
from resnet.cifar10.models import resnet, densenet, resnet_single_channel
from resnet.cifar10.datasets import CoarseCIFAR100
from resnet.cifar10.all_models import MODELS

from resnet.cifar10.trainer import Trainer

@click.group(invoke_without_command=True)
@click.option('--dataset-dir', default='./data')
@click.option('--checkpoint', '-c', type=click.Choice(['best', 'all', 'last']),
              default='last')
@click.option('--restore', '-r', is_flag=True)
@click.option('--tracking/--no-tracking', default=True)
@click.option('--track-test-acc/--no-track-test-acc', default=True)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--epochs', '-e', multiple=True, default=[200],
              type=int)
@click.option('--batch-size', '-b', default=32)
@click.option('--learning-rates', '-l', multiple=True, default=[1e-3],
              type=float)
@click.option('--momentum', default=0.9)
@click.option('--optimizer', '-o', type=click.Choice(['sgd', 'adam']),
              default='sgd')
@click.option('--schedule/--no-schedule', default=False)
@click.option('--patience', default=10)
@click.option('--decay-factor', default=0.1)
@click.option('--min-lr', default=1e-7)
@click.option('--augmentation/--no-augmentation', default=True)
@click.option('device_ids', '--device', '-d', multiple=True, type=int)
@click.option('--num-workers', type=int)
@click.option('--weight-decay', default=1e-4)
@click.option('--validation', '-v', default=0.0)
@click.option('--evaluate', is_flag=True)
@click.option('--shuffle/--no-shuffle', default=True)
@click.option('--half', is_flag=True)
@click.option('--arch', '-a', type=click.Choice(MODELS.keys()),
              default='resnet20')
@click.pass_context
def train(ctx, dataset_dir, checkpoint, restore, tracking, track_test_acc,
          cuda, epochs, batch_size, learning_rates, momentum, optimizer,
          schedule, patience, decay_factor, min_lr, augmentation,
          device_ids, num_workers, weight_decay, validation, evaluate, shuffle,
          half, arch):
    timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
    local_timestamp = str(datetime.now())  # noqa: F841

    dataset = ctx.obj['dataset'] if ctx.obj is not None else 'cifar10'
    assert dataset in data.DATASETS, "Only CIFAR and SVHN supported"

    dataset = data.DATASETS[dataset]
    dataset_dir = dataset.create_path(dataset_dir)
    num_classes = dataset.num_classes

    ### Use single-channel ResNet for MNIST, default is 3-channel ###
    if dataset == 'mnist' and 'resnet' in arch:
        arch += '-1'
        
    config = {k: v for k, v in locals().items() if k != 'ctx'}

    if ctx.invoked_subcommand is not None:
        click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        for k, v in config.items():
            ctx.obj[k] = v
        return 0

    learning_rate = learning_rates[0]
    use_cuda = cuda and torch.cuda.is_available()

    if use_cuda:
        device_ids = device_ids or list(range(torch.cuda.device_count()))
        num_workers = num_workers or len(device_ids)
    else:
        num_workers = num_workers or 1
    print(f"using {num_workers} workers for data loading")

    # load data
    print("Preparing {} data:".format(dataset.name.upper()))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset.mean, dataset.std)
    ])

    test_dataset = dataset.create_test_dataset(dataset_dir, transform_test)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=use_cuda)

    if evaluate:
        print("Only running evaluation of model on test dataset")
        run(start_epoch - 1, model, test_loader, use_cuda=use_cuda,
            tracking=test_results_file, train=False)
        return

    if augmentation:
        transform_train = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
        ]
    else:
        transform_train = []

    transform_train = transforms.Compose(transform_train + [
        transforms.ToTensor(),
        transforms.Normalize(dataset.mean, dataset.std),
    ])

    train_dataset = dataset.create_train_dataset(dataset_dir, transform_train)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    assert 1 > validation and validation >= 0, "Validation must be in [0, 1)"
    split = num_train - int(validation * num_train)

    if shuffle:
        np.random.shuffle(indices)

    train_indices = indices[:split]
    valid_indices = indices[split:]

    print('Using {} examples for training'.format(len(train_indices)))
    print('Using {} examples for validation'.format(len(valid_indices)))

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_cuda)
    if validation != 0:
        valid_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=valid_sampler, batch_size=batch_size,
            num_workers=num_workers, pin_memory=use_cuda)
    else:
        print('Using test dataset for validation')
        valid_loader = test_loader

    if not restore:
        trainer = Trainer.create_new(arch, num_classes, optimizer,
                                     list(zip(epochs, learning_rates)),
                                     momentum=momentum, weight_decay=weight_decay,
                                     use_cuda=use_cuda, basedir='./run')
    else:
        trainer = Trainer.from_checkpoint(arch, num_classes, optimizer,
                                          file_utils.load_latest_checkpoint('./run', arch),
                                          list(zip(epochs, learning_rates)),
                                          use_cuda=use_cuda)
                                          
    trainer.train_loop(train_loader, valid_loader)
    
if __name__ == '__main__':
    train()
