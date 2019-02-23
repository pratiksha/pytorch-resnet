'''
Gaussian certification per https://arxiv.org/pdf/1902.02918.pdf
'''

import torch
import numpy as np

from torch.distributions import Normal
from torch.autograd import Variable

from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint

from resnet.cifar10.trainer import TrainingState

class Certifier(object):
    @staticmethod
    def from_checkpoint(checkpoint_dir, sigma, n0, n, alpha, num_classes):
        restored_state = TrainingState.from_checkpoint(checkpoint_dir.best_checkpoint_path())
        return Certifier(restored_state.model,
                         sigma, n0, n, alpha, num_classes)
    
    '''
    Variables per the paper.
    f: base classifier
    sigma_2: variance of Gaussian noise for sampling distribution
    n0: number of samples for initial guess of top class
    n: number of samples for bound
    alpha: error probability
    '''
    def __init__(self, f, sigma, n0, n, alpha, num_classes):
        self.f = f
        self.f.cuda()
        
        self.sigma = sigma
        self.n0 = n0
        self.n = n
        self.alpha = alpha
        self.num_classes = num_classes

    @staticmethod
    def expand(v, shape):
        return torch.Tensor([v]).expand(shape)

    '''
    Take nsamples samples around x and output the predictions of f on those samples.
    '''
    def sample_under_noise(self, nsamples, x):
        x = x.cuda()
        stds = Certifier.expand(self.sigma, [nsamples] + list(x.shape)[1:]) # exclude minibatch dimension

        start = 0
        batch = 1000
        counts = torch.zeros(self.num_classes)

        while start < nsamples:
            end = min(nsamples, start+batch)

            size = torch.Size(tuple([end-start] + list(x.shape)[1:]))
            epsilons = torch.cuda.FloatTensor(size).normal_(0, self.sigma)
            samples = x.expand_as(epsilons) + epsilons

            v_samples = Variable(samples, requires_grad=False, volatile=True)
            v_samples = v_samples.cuda()
            outputs = self.f(v_samples)
            _, predictions = torch.max(outputs.data, 1)
            for c in predictions:
                counts[c] += 1 # TODO there must be a better way to do this...
            start += batch
        
        return counts

    '''
    Evaluate g (smoothed f) at x
    '''
    def predict(self, x):
        counts = self.sample_under_noise(self.n, x)
        class_counts, class_idxs = torch.topk(counts, k=2, dim=0)

        chat_A = class_idxs[0]
        chat_B = class_idxs[1]
        n_A = class_counts[0]
        n_B = class_counts[1]

        ## if the test says the difference between A and B is significant...
        pval = binom_test(n_A, n=n_A + n_B)
        if pval < self.alpha:
            ## return the top class
            return chat_A
        
        ## else abstain
        return -1
        
    
    '''
    Get a certification radius for top class label
    '''
    def certify(self, x):
        guess_counts = self.sample_under_noise(self.n0, x)
        g_count_A, g_chat_A = torch.topk(guess_counts, k=1)
        counts = self.sample_under_noise(self.n, x)
        pA_low, pA_hi = proportion_confint(counts[g_chat_A], self.n, 1-self.alpha)
        if pA_low > 0.5:
            return (g_chat_A, self.sigma * norm.ppf(pA_low))
        else:
            # abstain!
            return (-1, -1)
