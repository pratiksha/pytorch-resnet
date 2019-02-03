'''
Utilities to train a majority-vote ensemble, where each model in the ensemble can be trained and tested independently.
'''

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

from resnet import utils
from resnet.cifar10.models import resnet, densenet
from resnet.cifar10.datasets import CoarseCIFAR100
from resnet.cifar10 import subsample_transform

from resnet.cifar10.train import DATASETS, MEANS, STDS, MEANSTDS, SHAPES, MODELS, correct
from resnet.cifar10.ensemble_dataset import EnsembleDataset

def run(epoch, model_infos, loader, criterion=None, top=(1, 5),
        use_cuda=False, tracking=None, train=True, half=False):
    accuracies = [utils.AverageMeter() for _ in top]

    assert criterion is not None or not train, 'Need criterion to train model'

    loader = tqdm.tqdm(loader)
    if train:
        losses = utils.AverageMeter()
    
    for mi in model_infos:
        if train:
            mi.model.train()
        else:
            mi.model.eval()

    start = datetime.now()
    for batch_index, data_batch in enumerate(loader):
        all_inputs, all_targets = list(zip(*data_batch)) ## Note that the targets should all be the same.
        all_inputs  = [Variable(inp, requires_grad=False, volatile=not train) for inp in all_inputs]
        all_targets = [Variable(targ, requires_grad=False, volatile=not train) for targ in all_targets]
        batch_size = all_targets[0].size(0)
        assert batch_size < 2**32, 'Size is too large! correct will overflow'

        if use_cuda:
            all_inputs = [x.cuda() for x in all_inputs]
            all_targets = [x.cuda() for x in all_targets]
            if half:
                all_inputs = [x.half() for x in all_inputs]

        all_outputs = []
        ## Optimize each model independently.
        for mi, inputs, targets in zip(model_infos, all_inputs, all_targets):
            outputs = mi.model(inputs)
            all_outputs.append(outputs.clone())

            if train:
                loss = criterion(outputs, targets)
                mi.optimizer.zero_grad()
                loss.backward()
                mi.optimizer.step()
                losses.update(loss.data[0], batch_size)

        ## here is where the majority vote goes
        acc_outputs = all_outputs[0]
        for x in all_outputs[1:]:
            acc_outputs = torch.add(acc_outputs, 1, x) # for now just add all the output vectors together and take max as "majority"
        # re-normalize
        acc_outputs = acc_outputs.div_(len(all_outputs))
            
        _, predictions = torch.max(acc_outputs.data, 1)
        top_correct = correct(acc_outputs, all_targets[0], top=top) ## again, all targets should be the same
        for i, count in enumerate(top_correct):
            accuracies[i].update(count * (100. / batch_size), batch_size)

        end = datetime.now()
        if tracking is not None:
            result = OrderedDict()
            result['timestamp'] = datetime.now()
            result['batch_duration'] = end - start
            result['epoch'] = epoch
            result['batch'] = batch_index
            result['batch_size'] = batch_size
            for i, k in enumerate(top):
                result['top{}_correct'.format(k)] = top_correct[i]
                result['top{}_accuracy'.format(k)] = accuracies[i].val
            if train:
                result['loss'] = loss.data[0]
            utils.save_result(result, tracking)

        desc = 'Epoch {} {}'.format(epoch, '(Train):' if train else '(Val):  ')
        if train:
            desc += ' Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)
        for k, acc in zip(top, accuracies):
            desc += ' Prec@{} {acc.val:.3f} ({acc.avg:.3f})'.format(k, acc=acc)
        loader.set_description(desc)
        start = datetime.now()

    if train:
        message = 'Training accuracy of'
    else:
        message = 'Validation accuracy of'
    for i, k in enumerate(top):
        message += ' top-{}: {}'.format(k, accuracies[i].avg)
    print(message)
    return accuracies[0].avg


# Index specifies the index into the ensemble.
def create_graph(arch, timestamp, index, optimizer, restore,
                 learning_rate=None,
                 momentum=None,
                 weight_decay=None, num_classes=10):
    # create model
    model = MODELS[arch](num_classes=num_classes)

    # create optimizer
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(optimizer))

    if restore is not None:
        if restore == 'latest':
            restore = utils.latest_file(arch) + '/' + str(index)
        print(f'Restoring model from {restore}')
        assert os.path.exists(restore)
        restored_state = torch.load(restore)
        assert restored_state['arch'] == arch

        model.load_state_dict(restored_state['model'])

        if 'optimizer' in restored_state:
            optimizer.load_state_dict(restored_state['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = learning_rate

        run_dir = os.path.split(restore)[0]
    else:
        run_dir = f"./run/{arch}/{timestamp}/{index}"

    if not os.path.exists(run_dir) and run_dir != '':
        os.makedirs(run_dir)

    print(model)
    print("{} parameters".format(utils.count_parameters(model)))
    print(f"Run directory set to {run_dir}")

    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))
    return run_dir, model, optimizer

class ModelInfo:
    def __init__(self, run_dir, model, optimizer):
        self.run_dir = run_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = None

class Ensemble:
    def __init__(self, run_dir, start_epoch, models, timestamp, best_accuracy = 0.0):
        self.run_dir = run_dir
        self.start_epoch = start_epoch
        self.models = models # list of ModelInfo
        self.timestamp = timestamp
        self.best_accuracy = best_accuracy

# restore is top-level directory of restore directories for this model
def create_ensemble(arch, timestamp, optimizer, restore,
                    learning_rate=None,
                    momentum=None,
                    weight_decay=None,
                    num_classes=10,
                    num_models=1):

    start_accuracy = 0.0
    start_epoch = 1
    if restore is not None:
        if restore == 'latest':
            restore = utils.latest_file(arch) + '/' + str(index)
        print(f'Restoring ensemble from {restore}')
        assert os.path.exists(restore)
        restored_state = torch.load(restore)

        start_accuracy = restored_state['accuracy']
        start_epoch = restored_state['epoch'] + 1

    print('Starting accuracy is {}'.format(start_accuracy))

    models = []
    for i in range(num_models):
        model_restore = restore
        if model_restore is not None and model_restore != 'latest':
            model_restore += '/' + str(i)
            
        run_dir, model, opt_obj = create_graph(
            arch, timestamp, i, optimizer, model_restore,
            learning_rate=learning_rate, momentum=momentum,
            weight_decay=weight_decay, num_classes=num_classes)
        models.append(ModelInfo(run_dir, model, opt_obj))

    run_dir = f"./run/{arch}/{timestamp}"
    
    ensemble = Ensemble(run_dir, start_epoch, models, timestamp, start_accuracy)
    return ensemble
    
def create_test_dataset(dataset, dataset_dir, transform,
                        target_transform=None):
    if dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root=dataset_dir, train=False,
                                        download=True,
                                        transform=transform,
                                        target_transform=target_transform)
    elif dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(root=dataset_dir, train=False,
                                         download=True,
                                         transform=transform,
                                         target_transform=target_transform)
    elif dataset == 'cifar20':
        test_dataset = CoarseCIFAR100(root=dataset_dir, train=False,
                                      download=True, transform=transform,
                                      target_transform=target_transform)
    elif dataset == 'svhn' or dataset == 'svhn+extra':
        test_dataset = datasets.SVHN(root=dataset_dir, split='test',
                                     download=True,
                                     transform=transform,
                                     target_transform=target_transform)
    elif dataset == 'mnist':
        test_dataset = datasets.MNIST(root=dataset_dir, train=False,
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    return test_dataset


def create_train_dataset(dataset, dataset_dir, transform,
                         target_transform=None):
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=dataset_dir, train=True,
                                         download=True,
                                         transform=transform,
                                         target_transform=target_transform)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=dataset_dir, train=True,
                                          download=True,
                                          transform=transform,
                                          target_transform=target_transform)
    elif dataset == 'cifar20':
        train_dataset = CoarseCIFAR100(root=dataset_dir, train=True,
                                       download=True, transform=transform,
                                       target_transform=target_transform)
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN(root=dataset_dir, split='train',
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    elif dataset == 'svhn+extra':
        _train_dataset = datasets.SVHN(root=dataset_dir, split='train',
                                       download=True,
                                       transform=transform,
                                       target_transform=target_transform)
        _extra_dataset = datasets.SVHN(root=dataset_dir, split='extra',
                                       download=True,
                                       transform=transform,
                                       target_transform=target_transform)
        train_dataset = torch.utils.data.ConcatDataset([
            _train_dataset,
            _extra_dataset
        ])
    elif dataset == 'mnist':
        train_dataset = datasets.MNIST(root=dataset_dir, train=True,
                                       download=True,
                                       transform=transform,
                                       target_transform=target_transform)

    return train_dataset

@click.group(invoke_without_command=True)
@click.option('--dataset-dir', default='./data')
@click.option('--checkpoint', '-c', type=click.Choice(['best', 'all', 'last']),
              default='last')
@click.option('--restore', '-r')
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
def train_ensemble(ctx, dataset_dir, checkpoint, restore, tracking, track_test_acc,
                   cuda, epochs, batch_size, learning_rates, momentum, optimizer,
                   schedule, patience, decay_factor, min_lr, augmentation,
                   device_ids, num_workers, weight_decay, validation, evaluate, shuffle,
                   half, arch):
    timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
    local_timestamp = str(datetime.now())  # noqa: F841
    dataset = ctx.obj['dataset'] if ctx.obj is not None else 'cifar10'
    assert dataset in DATASETS, "Only CIFAR and SVHN supported"

    if dataset == 'svhn+extra':
        dataset_dir = os.path.join(dataset_dir, 'svhn')
    elif dataset == 'cifar20':
        dataset_dir = os.path.join(dataset_dir, 'cifar100')
    else:
        dataset_dir = os.path.join(dataset_dir, dataset)

    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'cifar20':
        num_classes = 20
    else:
        num_classes = 10
    config = {k: v for k, v in locals().items() if k != 'ctx'}

    if ctx.invoked_subcommand is not None:
        click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        for k, v in config.items():
            ctx.obj[k] = v
        return 0

    learning_rate = learning_rates[0]

    use_cuda = cuda and torch.cuda.is_available()

    # create dataset transforms
    stride = 2 # currently assumes that stride divides h and w evenly
    sample_transforms = []
    for i in range(stride**2):
        sample_transforms.append(subsample_transform.Subsample(stride, i)) # select ith pixel in stride x stride grid
        
    # one model per subsampled dataset
    ensemble = create_ensemble(arch, timestamp, optimizer,
                               restore, learning_rate=learning_rate,
                               momentum=momentum, weight_decay=weight_decay,
                               num_classes=num_classes,
                               num_models=len(sample_transforms))

    utils.save_config(config, ensemble.run_dir)

    if tracking:
        train_results_file = os.path.join(ensemble.run_dir, 'train_results.csv')
        valid_results_file = os.path.join(ensemble.run_dir, 'valid_results.csv')
        test_results_file = os.path.join(ensemble.run_dir, 'test_results.csv')
    else:
        train_results_file = None
        valid_results_file = None
        test_results_file = None

    # create loss
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        print('Copying ensemble to GPU')
        criterion = criterion.cuda()
        criterion = criterion.half()

        for mi in ensemble.models:
            mi.model = mi.model.cuda()
            
            if half:
                mi.model = mi.model.half()
            device_ids = device_ids or list(range(torch.cuda.device_count()))
            mi.model = torch.nn.DataParallel(
                mi.model, device_ids=device_ids)
            num_workers = num_workers or len(device_ids)
    else:
        num_workers = num_workers or 1
        if half:
            print('Half precision (16-bit floating point) only works on GPU')
    print(f"using {num_workers} workers for data loading")

    # load data
    print("Preparing {} data:".format(dataset.upper()))
    test_datasets = []
    for t in sample_transforms:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEANS[dataset], STDS[dataset]),
            t,
        ])

        test_datasets.append(create_test_dataset(dataset, dataset_dir, transform_test))
    test_ensemble = EnsembleDataset(test_datasets)
        
    test_loader = torch.utils.data.DataLoader(
        test_ensemble,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=use_cuda)

    if evaluate:
        print("Only running evaluation of model on test dataset")
        run(ensemble.start_epoch - 1, ensemble.models, test_loader, use_cuda=use_cuda,
            tracking=test_results_file, train=False)
        return

    # TODO removing augmentation for ensembles, maybe add back in later
    # if augmentation:
    #     transform_train = [
    #             transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip()
    #     ]
    # else:
    #     transform_train = []

    train_datasets = []
    for t in sample_transforms:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEANS[dataset], STDS[dataset]),
            t,
        ])
        
        train_datasets.append(create_train_dataset(dataset, dataset_dir, transform_train))
    ensemble_train = EnsembleDataset(train_datasets)
        
    num_train = len(train_datasets[0])
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
        ensemble_train, sampler=train_sampler, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_cuda)
    if validation != 0:
        valid_loader = torch.utils.data.DataLoader(
            ensemble_train, sampler=valid_sampler, batch_size=batch_size,
            num_workers=num_workers, pin_memory=use_cuda)
    else:
        print('Using test dataset for validation')
        valid_loader = test_loader

    for nepochs, learning_rate in zip(epochs, learning_rates):
        end_epoch = ensemble.start_epoch + nepochs

        for mi in ensemble.models:
            for group in mi.optimizer.param_groups:
                group['lr'] = learning_rate
            _lr_optimizer = utils.get_learning_rate(mi.optimizer)
            if _lr_optimizer is not None:
                print('Learning rate set to {}'.format(_lr_optimizer))
                assert _lr_optimizer == learning_rate

            if schedule:
                mi.scheduler = ReduceLROnPlateau(
                    mi.optimizer, 'max', factor=decay_factor,
                    verbose=True, patience=patience)

        for epoch in range(ensemble.start_epoch, end_epoch):
            run(epoch, ensemble.models, train_loader, criterion,
                use_cuda=use_cuda, tracking=train_results_file, train=True,
                half=half)

            valid_acc = run(epoch, ensemble.models, valid_loader, use_cuda=use_cuda,
                            tracking=valid_results_file, train=False,
                            half=half)

            if validation != 0 and track_test_acc:
                run(epoch, ensemble.models, test_loader, use_cuda=use_cuda,
                    tracking=test_results_file, train=False)

            for mi in ensemble.models:
                if schedule:
                    prev_learning_rate = utils.get_learning_rate(mi.optimizer)
                    mi.scheduler.step(valid_acc)
                    new_learning_rate = utils.get_learning_rate(mi.optimizer)

                    if new_learning_rate <= min_lr:
                        return 0

                    if prev_learning_rate != new_learning_rate:
                        timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
                        config['timestamp'] = timestamp
                        config['learning_rate'] = new_learning_rate
                        config['next_epoch'] = epoch + 1
                        utils.save_config(config, mi.run_dir)

            is_best = valid_acc > ensemble.best_accuracy
            last_epoch = epoch == (end_epoch - 1)
            if is_best or checkpoint == 'all' or (checkpoint == 'last' and last_epoch):  # noqa: E501
                ensemble_state = {
                    'epoch': epoch,
                    'accuracy': valid_acc
                }
            if is_best:
                print('New best ensemble!')
                filename = os.path.join(ensemble.run_dir, 'checkpoint_best_model.t7')
                print(f'Saving checkpoint to {filename}')
                ensemble.best_accuracy = valid_acc
                torch.save(ensemble_state, filename)
            if checkpoint == 'all' or (checkpoint == 'last' and last_epoch):
                filename = os.path.join(ensemble.run_dir, f'checkpoint_{epoch}.t7')
                print(f'Saving checkpoint to {filename}')
                torch.save(ensemble_state, filename)
            
            
            for i in range(len(ensemble.models)):
                mi = ensemble.models[i]
                model = mi.model
                if is_best or checkpoint == 'all' or (checkpoint == 'last' and last_epoch):  # noqa: E501
                    state = {
                        'arch': arch,
                        'model': (model.module if use_cuda else model).state_dict(),  # noqa: E501
                        'optimizer': mi.optimizer.state_dict()
                    }
                if is_best:
                    print('New best ensemble!')
                    filename = os.path.join(ensemble.run_dir + '/' + str(i), 'checkpoint_best_model.t7')
                    print(f'Saving checkpoint to {filename}')
                    ensemble.best_accuracy = valid_acc
                    torch.save(state, filename)
                if checkpoint == 'all' or (checkpoint == 'last' and last_epoch):
                    filename = os.path.join(ensemble.run_dir + '/' + str(i), f'checkpoint_{epoch}.t7')
                    print(f'Saving checkpoint to {filename}')
                    torch.save(state, filename)

        ensemble.start_epoch = end_epoch


if __name__ == '__main__':
    train_ensemble()
