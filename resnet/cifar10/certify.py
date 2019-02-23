'''
Robust prediction & certification from Cohen et al. https://arxiv.org/pdf/1902.02918.pdf
'''


import os
from collections import OrderedDict

from datetime import datetime

import tqdm
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
from resnet.cifar10.gaussian_certifier import Certifier

'''
Get a certification radius for all examples in the dataset for the given model.
'''
@click.group(invoke_without_command=True)
@click.option('--dataset-dir', default='./data')
@click.option('--checkpoint-name', '-c')
@click.option('--restore', '-r', is_flag=True)
@click.option('--cuda/--no-cuda', default=True)
@click.option('device_ids', '--device', '-d', multiple=True, type=int)
@click.option('--num-workers', type=int)
@click.option('--batch-size', '-b', default=1)
@click.option('--arch', '-a', type=click.Choice(MODELS.keys()),
              default='resnet20')
@click.option('--alpha', '-l', type=float,
              default=0.001)
@click.pass_context
def certify(ctx, dataset_dir, checkpoint_name, restore,
            batch_size,
            cuda, device_ids, num_workers, arch,
            alpha):

    n0 = 100
    n = 100000
    sigma = 0.5
    
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


    checkpoint = file_utils.load_checkpoint('./run', arch, checkpoint_name)
    certifier = Certifier.from_checkpoint(checkpoint, sigma, n0, n, alpha, num_classes)

    avg_radius = utils.AverageMeter()
    min_radius = 100000
    abstentions = 0

    start = datetime.now()
    loader = tqdm.tqdm(test_loader)
    radii = []
    for i, (input, target) in enumerate(loader):
        class_A, radius = certifier.certify(input)
        if radius < 0:
            abstentions += 1
        else:
            avg_radius.update(radius)
            min_radius = min(radius, min_radius)
            radii.append(radius)
        desc = "Certified radius: {}".format(radius)
        loader.set_description(desc)

    end = datetime.now()
    
    print('Final min certified radius: {}, time: {}'.format(min_radius, end-start))
    with open('radii.txt', 'w') as f:
        f.write(str(radii))

if __name__=='__main__':
    certify()

