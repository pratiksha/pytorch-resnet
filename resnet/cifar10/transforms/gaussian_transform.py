'''
Apply Gaussian noise to an image
'''

import torch

from numbers import Number

class GaussianPerturb(object):
    def __init__(self, sigma):
        self.mean = 0
        self.std = sigma

    def __call__(self, image):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(image.shape)
            else:
                raise

        return image + torch.normal(expand(self.mean), expand(self.std))
