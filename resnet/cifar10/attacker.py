import torch
import tqdm

from datetime import datetime

from torch import nn
from torch.autograd import Variable

from resnet import utils

from resnet.cifar10.tester import Tester
from resnet.cifar10.trainer import TrainingState

from resnet.cifar10.attacks.attack_iterative import AttackIterative

class Attacker(Tester):
    @staticmethod
    def from_checkpoint(arch, num_classes, restore_dir, use_cuda=True):
        restored_state = TrainingState.from_checkpoint(restore_dir.best_checkpoint_path())
        attacker = Attacker(restored_state.model,
                            nn.CrossEntropyLoss(),
                            use_cuda)
        return attacker

    def __init__(self, model, loss_criterion, use_cuda,
                 attack_name='iterative'):
        super().__init__(model, loss_criterion, use_cuda)
        
        self.attack = AttackIterative(norm=2, debug=False)

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
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs_adv = self.attack.run(self.model, inputs, targets, batch_index)

            outputs, top1_acc = self.evaluate_model(inputs_adv, targets, volatile=False)
            _, pred = torch.max(outputs.data, 1)
            acc.update(top1_acc)
            
            end = datetime.now()

            desc = ' Prec@{} {acc.val:.3f} ({acc.avg:.3f}), output {pred} target {targets}'.format(1, acc=acc, pred=pred[0], targets=targets[0])
            loader.set_description(desc)

            start = datetime.now()

        message = 'Adversarial accuracy of'
        message += ' top-{}: {}'.format(1, acc.avg)
        print(message)

        return acc.avg
