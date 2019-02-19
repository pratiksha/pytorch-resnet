import os
from collections import OrderedDict

import click
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets

from resnet import utils, file_utils, data
from resnet.cifar10.models import resnet, densenet, resnet_single_channel
from resnet.cifar10.datasets import CoarseCIFAR100
from resnet.cifar10.all_models import MODELS

from resnet.cifar10.tester import Tester

@click.group(invoke_without_command=True)
@click.option('--dataset-dir', default='./data')
@click.option('--checkpoint', '-c', type=click.Choice(['best', 'all', 'last']),
              default='last')
@click.option('--restore', '-r', is_flag=True)
@click.option('--cuda/--no-cuda', default=True)
@click.option('device_ids', '--device', '-d', multiple=True, type=int)
@click.option('--num-workers', type=int)
@click.option('--batch-size', '-b', default=32)
@click.option('--arch', '-a', type=click.Choice(MODELS.keys()),
              default='resnet20')
@click.pass_context
def evaluate(ctx, dataset_dir, checkpoint, restore,
             batch_size,
             cuda, device_ids, num_workers, arch):
    dataset = ctx.obj['dataset'] if ctx.obj is not None else 'cifar10'
    assert dataset in data.DATASETS, "Only CIFAR and SVHN supported"

    dataset = data.DATASETS[dataset]
    dataset_dir = dataset.create_path(dataset_dir)
    num_classes = dataset.num_classes

    ### Use single-channel ResNet for MNIST, default is 3-channel ###
    if dataset == 'mnist' and 'resnet' in arch:
        arch += '-1'

    if ctx.invoked_subcommand is not None:
        click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        for k, v in config.items():
            ctx.obj[k] = v
        return 0

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

    tester = Tester.from_checkpoint(arch, num_classes, 
                                    file_utils.load_latest_checkpoint('./run', arch),
                                    use_cuda=use_cuda)

    acc = tester.run_eval(test_loader)
    print(f"Test accuracy: {acc}")
    
if __name__ == '__main__':
    evaluate()
