import torch
import tqdm
import numpy as np

from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import datasets

from resnet import file_utils
from resnet.file_utils import ModelState, OptimizerState
from resnet.cifar10.all_models import MODELS

class OptimizerState(object):
    @staticmethod
    def init_optimizer(model, optimizer_name, lr, momentum=None, weight_decay=None):
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            return optim.SGD(model.parameters(), lr=learning_rate,
                             momentum=momentum,
                             weight_decay=weight_decay)
        else:
            raise NotImplementedError("Unknown optimizer: {}".format(optimizer))

    @staticmethod
    def from_restore(restored_state, model):
        assert model is not None # have to pass in associated model so we can initialize with its parameters

        momentum = 'momentum' in restored_state.keys() ? restored_state['momentum'] : None
        weight_decay = 'weight_decay' in restored_state.keys() ? restored_state['weight_decay'] : None
        optimizer = init_optimizer(model, restored_state['name'],
                                   restored_state['lr'],
                                   restored_state['weight_decay'])

        return OptimizerState(restored_state['name'], optimizer, restored_state['lr'],
                              momentum, weight_decay)

    @staticmethod
    def new_state(model, optimizer_name, lr, momentum=None, weight_decay=None):
        optimizer = init_optimizer(model, optimizer_name, lr, momentum, weight_decay)
        return OptimizerState(optimizer_name, optimizer, lr, momentum, weight_decay)
    
    def __init__(self, name, optimizer, lr, momentum=None, weight_decay=None):
        self.name = name
        self.optimizer = optimizer
        self.learning_rate = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def dump_state(self):
        state_dict = {'name':self.name,
                      'optimizer':self.optimizer.state_dict(),
                      'lr': self.learning_rate,
                      'momentum': self.momentum,
                      'weight_decay': self.weight_decay}

class TrainingState(object):
    @staticmethod
    def from_restore(restored_state):
        arch = restored_state['arch']
        num_classes = restored_state['num_classes']
        model = MODELS[arch](num_classes=num_classes)
        model.load_state_dict(restored_state['model'])

        optimizer_name = restored_state['optimizer_name']
        optimizer_state = OptimizerState.from_restore(restored_state['optimizer_state'],
                                                      model)
        
        return TrainingState(model,
                             num_classes,
                             restored_state['epoch'] + 1,
                             arch,
                             restored_state['accuracy'],
                             restored_state['optimizer_name'],
                             optimizer_state)
    
    @staticmethod
    def from_checkpoint(ckpt_path):
        restored_state = torch.load(ckpt_path)
        return from_restore(restored_state)

    @staticmethod
    def new_state(self, model, num_classes, arch, optimizer_name, optimizer_state):
        return TrainingState(model, num_classes, 0, arch, 0, optimizer_name, optimizer_state)
    
    def __init__(self, model, num_classes, epoch, arch, acc, optimizer_name, optimizer_state):
        self.model = model
        self.num_classes = num_classes
        self.epoch = epoch
        self.arch = arch
        self.cur_valid_acc = None # We assume we're starting at this epoch, so we shouldn't associate the epoch with any acc until the epoch completes.
        self.best_acc = acc
        self.optimizer_name = optimizer_name
        self.optimizer_state = optimizer_state # this is an OptimizerState

    def dump_state(self):
        optimizer_state = self.optimizer_state.dump_state()
        state_dict = {
            'model': (self.model.module if use_cuda else self.model).state_dict(),
            'num_classes': self.num_classes,
            'arch': self.arch,
            'epoch': self.epoch,
            'accuracy': self.valid_acc,
            'optimizer_name': self.optimizer_name,
            'optimizer_state': optimizer_state,
        }
        
        return state_dict

    def model(self):
        return self.model
    
    def optimizer(self):
        return self.optimizer_state.optimizer
    
class Trainer:

    @staticmethod
    def from_state(arch, num_classes, optimizer_name, restore_dir):
        restored_state = TrainingState.from_checkpoint(restore_dir.latest_checkpoint_path())
        trainer = Trainer(model, optimizer, restore_dir, restored_state)

    @staticmethod
    def create_new(arch, num_classes, optimizer_name):
        pass
        
    '''
    initial_state is a TrainingState
    checkpoint_dir is a CheckpointDir
    '''
    def __init__(self, initial_state, checkpoint_dir, decay_factor=0.1, patience=10):
        self.config = None

        self.state = initial_state # contains model, epoch, current best acc, last validation acc, etc.

        # TODO for tracking results, eventually
        self.train_res_file = None
        self.valid_res_file = None
        self.test_res_file = None

        # TODO this can/should also go in the training state
        if schedule:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 'max', factor=decay_factor,
                verbose=True, patience=patience)
        
    def dump_state(self):
        optimizer_state = OptimizerState(self.optimizer, self.optimizer,
                                         self.optimizer
        return ModelState(self.model, self.num_classes, self.arch, self.epoch,
                          self.cur_acc, self.optimizer_name)
        return {'epoch': self.epoch,
                'arch': self.arch,
                'model': self.model,
                'accuracy': self.cur_acc,
                'optimizer': self.optimizer.state_dict()}

    def save_checkpoint(self, best=False):
        state = self.dump_state()
        if best:
            ckpt_path = self.checkpoint_dir.best_checkpoint_path()
        else:
            ckpt_path = self.checkpoint_dir.epoch_path(self.epoch)

        torch.save(state, ckpt_path)

    def save_new_lr(self, epoch, lr): # TODO what should this actually do?
        timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
        config['timestamp'] = timestamp
        config['learning_rate'] = new_learning_rate
        config['next_epoch'] = epoch + 1
        utils.save_config(config, run_dir)
    
    def train(self):
        for (nepochs, learning_rate) in self.lr_schedule:
            end_epoch = self.epoch + nepochs
            self.run()

            valid_acc = self.run()

            if self.scheduler is not None:
                prev_lr = utils.get_learning_rate(self.optimizer)
                self.scheduler.step(valid_acc)
                new_lr = utils.get_learning_rate(self.optimizer)

                if new_learning_rate <= min_lr:
                    return 0

                if prev_learning_rate != new_learning_rate:
                    self.save_new_lr(epoch+1, new_learning_rate)

                is_best = valid_acc > self.best_acc
                checkpoint = is_best or (self.checkpoint=='all') or ((self.checkpoint == 'last') and (self.epoch == end_epoch - 1))
                
                if checkpoint:
                    if is_best:
                        print('New best model!')
                        self.save_checkpoint(best=True)
                    else:
                        self.save_checkpoint(best=False)
