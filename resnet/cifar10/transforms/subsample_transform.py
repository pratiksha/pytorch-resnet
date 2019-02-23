import torch

def create_mask(input_shape):
    prob_matrix = torch.ones(input_shape).div_(2) # 0.5 probability
    rand_mask = torch.bernoulli(prob_matrix)
    
    return rand_mask.byte()

class Mask(object):
    """
    Subsample pixels of an image, zeroing out the pixels not sampled 
    (i.e. apply a mask).
    
    Note that the subsample needs to be structured in order for convolution
    to still be a sensible operation on the image.
    """
    
    def __init__(self, output_mask):
        assert(isinstance(output_mask, torch.ByteTensor))
        self.output_mask = output_mask
        
    def __call__(self, image):
        # image is channels x h x w
        assert(image.shape[-2:] == self.output_mask.shape)
        assert(len(image.shape) == 3) # currently don't support more dimensions
        
        img = image.clone()
        for i in range(image.shape[0]):
            # zero within layer
            img[i][self.output_mask] = 0
        
        return img

class Subsample(object):
    """
    Subsample pixels of an image, reducing the image size.
    Current strategy: choose pixels based on stride and then
    consolidate into a square.
    
    Invariants: square image remains square after sampling.
    """
    
    def __init__(self, stride, index):
        # index is 0 <= i < stride**2 that selects which element of the
        #   stride x stride square is retained in the final image.

        assert(isinstance(stride, int))
        assert(isinstance(index, int))
        assert(index < stride**2)
        
        self.stride = stride
        self.row = (index // self.stride)
        self.col = index - (self.row)*self.stride
    
    def __call__(self, image):
        # image is channels x h x w
        
        # we assume the stride should be divisible evenly into both h and w
        assert(image.shape[-1] % self.stride == 0)
        assert(image.shape[-2] % self.stride == 0)
        assert(len(image.shape) == 3)
        
        # probably not the most efficient way to do this...
        downsampled_img = torch.zeros(torch.Size(((int)(image.shape[0]), 
                                      (int)(image.shape[-1] / self.stride),
                                      (int)(image.shape[-2] / self.stride))))
        for i in range(downsampled_img.shape[-1]):
            for j in range(downsampled_img.shape[-2]):
                downsampled_img[:, i, j] = image[:, i*self.stride + self.row, j*self.stride + self.col]
        
        return downsampled_img
