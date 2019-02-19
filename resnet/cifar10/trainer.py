import torch
import tqdm
import numpy as np

from datetime import datetime

from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import datasets

from resnet import utils, file_utils
from resnet.cifar10.all_models import MODELS

class OptimizerState(object):
    @staticmethod
    def init_optimizer(model, optimizer_name, lr, momentum=None, weight_decay=None):
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(model.parameters(), lr=lr,
                             momentum=momentum,
                             weight_decay=weight_decay)
        else:
            raise NotImplementedError("Unknown optimizer: {}".format(optimizer))

    @staticmethod
    def from_restore(restored_state, model):
        assert model is not None # have to pass in associated model so we can initialize with its parameters

        momentum = restored_state['momentum'] if 'momentum' in restored_state.keys() else None
        weight_decay = restored_state['weight_decay'] if 'weight_decay' in restored_state.keys() else None

        optimizer = OptimizerState.init_optimizer(model, restored_state['name'],
                                                  restored_state['lr'], momentum, weight_decay)

        return OptimizerState(restored_state['name'], optimizer, restored_state['lr'],
                              momentum, weight_decay)

    @staticmethod
    def new_state(model, optimizer_name, lr, momentum=None, weight_decay=None):
        optimizer = OptimizerState.init_optimizer(model, optimizer_name, lr, momentum, weight_decay)
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
        return state_dict

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
                             restored_state['best_accuracy'],
                             restored_state['optimizer_name'],
                             optimizer_state)
    
    @staticmethod
    def from_checkpoint(ckpt_path):
        restored_state = torch.load(ckpt_path)
        return TrainingState.from_restore(restored_state)

    @staticmethod
    def new_state(arch, num_classes, optimizer_name, lr, momentum=None, weight_decay=None):
        model = MODELS[arch](num_classes=num_classes)
        optimizer_state = OptimizerState.new_state(model, optimizer_name, lr, momentum, weight_decay)
        return TrainingState(model, num_classes, 0, arch, 0, optimizer_name, optimizer_state)
    
    def __init__(self, model, num_classes, epoch, arch, acc, optimizer_name, optimizer_state, use_cuda=False):
        self.model = model
        self.num_classes = num_classes
        self.epoch = epoch
        self.arch = arch
        self.best_acc = acc
        self.optimizer_name = optimizer_name
        self.optimizer_state = optimizer_state # this is an OptimizerState

        self.use_cuda = use_cuda

    def dump_state(self):
        optimizer_state = self.optimizer_state.dump_state()
        state_dict = {
            'model': (self.model.module if self.use_cuda else self.model).state_dict(),
            'num_classes': self.num_classes,
            'arch': self.arch,
            'epoch': self.epoch,
            'best_accuracy': self.best_acc,
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
    def from_checkpoint(arch, num_classes, optimizer_name, restore_dir, lr_schedule, use_cuda=True):
        # TODO restore latest?
        # restored_state = TrainingState.from_checkpoint(restore_dir.latest_checkpoint_path())
        restored_state = TrainingState.from_checkpoint(restore_dir.best_checkpoint_path())
        trainer = Trainer(restored_state,
                          nn.CrossEntropyLoss(),
                          lr_schedule,
                          restore_dir,
                          use_cuda)
        return trainer
        
    @staticmethod
    def create_new(arch, num_classes, optimizer_name, lr_schedule,
                   momentum=None, weight_decay=None, use_cuda=True, basedir='./run'):
        lr = lr_schedule[0][1]
        state = TrainingState.new_state(arch, num_classes, optimizer_name, lr, momentum, weight_decay)
        trainer = Trainer(state,
                          nn.CrossEntropyLoss(),
                          lr_schedule,
                          file_utils.new_checkpoint_dir(basedir, arch),
                          use_cuda)
        return trainer

    '''
    initial_state is a TrainingState
    checkpoint_dir is a CheckpointDir
    '''
    def __init__(self, initial_state, loss_criterion, lr_schedule, checkpoint_dir, use_cuda,
                 decay_factor=0.1, patience=10, schedule=False):
        self.state = initial_state # contains model, epoch, current best acc, last validation acc, etc.
        self.state.use_cuda = use_cuda
        self.checkpoint_dir = checkpoint_dir

        self.criterion = loss_criterion
        self.lr_schedule = lr_schedule

        if use_cuda:
            self.state.model.cuda()
            self.criterion.cuda()
            self.device_ids = list(range(torch.cuda.device_count()))
            self.state.model = torch.nn.DataParallel(
                self.state.model, device_ids=self.device_ids)
            self.num_workers = len(self.device_ids)
        else:
            num_workers = num_workers or 1

        # TODO for tracking results, eventually
        self.train_res_file = None
        self.valid_res_file = None
        self.test_res_file = None

        # TODO this can/should also go in the training state
        self.scheduler = None
        if schedule:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 'max', factor=decay_factor,
                verbose=True, patience=patience)
        
    def dump_state(self):
        return self.state.dump_state()

    def save_checkpoint(self, best=False):
        state = self.dump_state()
        
        if best:
            ckpt_path = self.checkpoint_dir.best_checkpoint_path()
        else:
            ckpt_path = self.checkpoint_dir.epoch_path(self.state.epoch)

        torch.save(state, ckpt_path)

    def save_new_lr(self, epoch, lr): # TODO what should this actually do?
        timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
        config['timestamp'] = timestamp
        config['learning_rate'] = new_learning_rate
        config['next_epoch'] = epoch + 1
        utils.save_config(config, run_dir)

    ## run model on inputs and evaluate against targets
    def evaluate_model(self, inputs_, targets_, volatile=False):
        inputs = Variable(inputs_, requires_grad=False, volatile=volatile)
        targets = Variable(targets_, requires_grad=False, volatile=volatile)

        batch_size = targets.size(0)
        assert batch_size < 2**32, 'Size is too large! correct will overflow'

        if self.state.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
        outputs = self.state.model(inputs)
        _, predictions = torch.max(outputs.data, 1)
        targets = targets.data.view_as(predictions)

        correct = predictions.eq(targets).cpu().int().sum(0)[0]
        return (outputs, correct * (100./batch_size))
        
    # since we are passing to optimizer, outputs and targets must both be Variables
    def gradient_update(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        self.state.optimizer().zero_grad()
        loss.backward()
        self.state.optimizer().step()

        return loss
        
    def run_train_iteration(self, loader):
        loader = tqdm.tqdm(loader)
        losses = utils.AverageMeter()
        acc = utils.AverageMeter() # top-1 only for now

        self.state.model.train()

        start = datetime.now()
        for batch_index, (inputs, targets) in enumerate(loader):
            outputs, top1_acc = self.evaluate_model(inputs, targets, volatile=False)
            targets_var = Variable(targets, requires_grad=False, volatile=False)

            if self.state.use_cuda:
                targets_var = targets_var.cuda()
            
            batch_size = targets.size(0)
            loss = self.gradient_update(outputs, targets_var)
            losses.update(loss.data[0], batch_size)

            acc.update(top1_acc)
            
            end = datetime.now()

            desc = 'Epoch {} {}'.format(self.state.epoch, '(Train):')
            desc += ' Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)
            desc += ' Prec@{} {acc.val:.3f} ({acc.avg:.3f})'.format(1, acc=acc)
            loader.set_description(desc)

            start = datetime.now()

        message = 'Training accuracy of'
        message += ' top-{}: {}'.format(1, acc.avg)
        print(message)

        return acc.avg

    def run_eval_iteration(self, loader):
        loader = tqdm.tqdm(loader)
        losses = utils.AverageMeter()
        acc = utils.AverageMeter() # top-1 only for now

        self.state.model.eval()

        start = datetime.now()
        for batch_index, (inputs, targets) in enumerate(loader):
            outputs, top1_acc = self.evaluate_model(inputs, targets, volatile=False)
            acc.update(top1_acc)
            
            end = datetime.now()

            desc = 'Epoch {} {}'.format(self.state.epoch, '(Val):  ')
            desc += ' Prec@{} {acc.val:.3f} ({acc.avg:.3f})'.format(1, acc=acc)
            loader.set_description(desc)

            start = datetime.now()

        message = 'Validation accuracy of'
        message += ' top-{}: {}'.format(1, acc.avg)
        print(message)

        return acc.avg

    def train_loop(self, train_loader, valid_loader):
        for (nepochs, learning_rate) in self.lr_schedule:  # lr_schedule is [(num_epochs, learning_rate)] list
            end_epoch = self.state.epoch + nepochs

            for group in self.state.optimizer().param_groups:
                group['lr'] = learning_rate
                _lr_optimizer = utils.get_learning_rate(self.state.optimizer())
                if _lr_optimizer is not None:
                    print('Learning rate set to {}'.format(_lr_optimizer))
                    assert _lr_optimizer == learning_rate
                    
            while self.state.epoch < end_epoch:
                train_acc = self.run_train_iteration(train_loader)
                valid_acc = self.run_eval_iteration(valid_loader)

                if self.scheduler is not None:
                    prev_lr = utils.get_learning_rate(self.state.optimizer())
                    self.scheduler.step(valid_acc)
                    new_lr = utils.get_learning_rate(self.state.optimizer())

                    if new_learning_rate <= min_lr:
                        return 0

                    if prev_learning_rate != new_learning_rate:
                        self.save_new_lr(epoch+1, new_learning_rate)

                is_best = valid_acc > self.state.best_acc
                checkpoint = is_best # TODO: or (self.checkpoint=='all') or ((self.checkpoint == 'last') and (self.epoch == end_epoch - 1))

                if is_best:
                    self.state.best_acc = valid_acc

                if checkpoint:
                    if is_best:
                        print('New best model!')
                        self.save_checkpoint(best=True)
                    else:
                        self.save_checkpoint(best=False)
                            
                self.state.epoch += 1
