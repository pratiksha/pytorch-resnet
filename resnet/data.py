'''
Utilities for dataset management (note: not the same as torch.Dataset, primarily handles metadata)
'''

import os

from torchvision import datasets

from resnet.cifar10.datasets import CoarseCIFAR100

class Data:
    def __init__(self, name, dataset, mean, std, shape, num_classes, num_channels):
        self.name = name
        
        self.pathname = name
        if self.pathname == 'svhn+extra':
            self.pathname = 'svhn'
        elif self.pathname == 'cifar20':
            self.pathname = 'cifar100'

        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.shape = shape
        self.num_classes = num_classes
        self.num_channels = num_channels

    def create_path(basedir):
        return os.path.join(basedir, self.name)

    def create_train_dataset(basedir, transform, target_transform=None):
        if self.name == 'svhn':
            return self.dataset(root=dataset_dir, split='train',
                                          download=True,
                                      transform=transform,
                                      target_transform=target_transform)
        elif self.name == 'svhn+extra':
            _train_dataset = self.dataset(root=dataset_dir, split='train',
                                          download=True,
                                          transform=transform,
                                          target_transform=target_transform)
            _extra_dataset = self.dataset(root=dataset_dir, split='extra',
                                          download=True,
                                          transform=transform,
                                          target_transform=target_transform)
            return torch.utils.data.ConcatDataset([
                _train_dataset,
                _extra_dataset
            ])
        else:
            return self.dataset(root=basedir, train=True,
                                download=True,
                                transform=transform,
                                target_transform=target_transform)

    
    def create_test_dataset(basedir, transform, target_transform=None):
        if self.name == 'svhn' or self.name == 'svhn+extra':
            return self.dataset(root=basedir, split='test',
                                download=True, transform=transform,
                                target_transform = target_transform)
        else:
            return self.dataset(root=basedir,
                                train=False,
                                download=True,
                                transform=transform,
                                target_transform=target_transform)
    
SVHN = Data('svhn',
            datasets.SVHN,
            (0.4377, 0.4438, 0.4728),
            (0.1201, 0.1231, 0.1052),
            None,
            10,
            3
)

CIFAR10 = Data('cifar10',
               datasets.CIFAR10,
               (0.4914, 0.4822, 0.4465),
               (0.24703223, 0.24348512, 0.26158784),,
               (3, 32, 32),
               10,
               3
)

CIFAR100 = Data('cifar100',
                datasets.CIFAR100,
                (0.5071, 0.4866, 0.4409),
                (0.26733428, 0.25643846, 0.27615047),
                None,
                100,
                3
)


CIFAR20 = Data('cifar20',
               CoarseCIFAR100,
               (0.5071, 0.4866, 0.4409),  # Same as CIFAR100
               (0.26733428, 0.25643846, 0.27615047),  # Same as CIFAR100
               None,
               20,
               3
)

SVHN_EXTRA = Data('svhn+extra',
                  datasets.SVHN,
                  (0.4309, 0.4302, 0.4463),
                  (0.1252, 0.1282, 0.1147),
                  None,
                  10,
                  3
)

MNIST = Data('mnist',
             datasets.MNIST,
             (0.1306,),
             (0.3015,),
             (1, 32, 32),
             10,
             1
)

DATASETS = {'mnist':MNIST,
            'svhn+extra':SVHN_EXTRA,
            'svhn':SVHN,
            'cifar100':CIFAR100,
            'cifar10':CIFAR10,
            'cifar20':CIFAR20,
}
