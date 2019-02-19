import torch
import tqdm

from datetime import datetime

from torch import nn
from torch.autograd import Variable

from resnet import utils
from resnet.cifar10.trainer import TrainingState

class Tester:
    @staticmethod
    def from_checkpoint(arch, num_classes, restore_dir, use_cuda=True):
        restored_state = TrainingState.from_checkpoint(restore_dir.best_checkpoint_path())
        tester = Tester(restored_state.model,
                        nn.CrossEntropyLoss(),
                        use_cuda)
        return tester
    
    def __init__(self, model, loss_criterion, use_cuda):
        self.criterion = loss_criterion
        self.model = model
        self.use_cuda = use_cuda

        if use_cuda:
            self.model.cuda()
            self.device_ids = list(range(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.device_ids)
            self.num_workers = len(self.device_ids)
        else:
            num_workers = num_workers or 1

    ## run model on inputs and evaluate against targets
    def evaluate_model(self, inputs_, targets_, volatile=False):
        inputs = Variable(inputs_, requires_grad=False, volatile=volatile)
        targets = Variable(targets_, requires_grad=False, volatile=volatile)

        batch_size = targets.size(0)
        assert batch_size < 2**32, 'Size is too large! correct will overflow'

        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
        outputs = self.model(inputs)
        _, predictions = torch.max(outputs.data, 1)
        targets = targets.data.view_as(predictions)

        correct = predictions.eq(targets).cpu().int().sum(0)[0]
        return (outputs, correct * (100./batch_size))

    def run_eval(self, loader):
        loader = tqdm.tqdm(loader)
        losses = utils.AverageMeter()
        acc = utils.AverageMeter() # top-1 only for now

        self.model.eval()

        start = datetime.now()
        for batch_index, (inputs, targets) in enumerate(loader):
            outputs, top1_acc = self.evaluate_model(inputs, targets, volatile=False)
            acc.update(top1_acc)
            
            end = datetime.now()

            desc = ' Prec@{} {acc.val:.3f} ({acc.avg:.3f})'.format(1, acc=acc)
            loader.set_description(desc)

            start = datetime.now()

        message = 'Validation accuracy of'
        message += ' top-{}: {}'.format(1, acc.avg)
        print(message)

        return acc.avg
