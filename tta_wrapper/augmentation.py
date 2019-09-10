import itertools
from . import functional as F


class Augmentation(object):

    transforms = {
        'h_flip': F.HFlip(),
        'v_flip': F.VFlip(),
        'rotation': F.Rotate(),
        'h_shift': F.HShift(),
        'v_shift': F.VShift(),
        'contrast': F.Contrast(),
        'add': F.Add(),
        'mul': F.Multiply(),
    }

    def __init__(self, **params):
        super().__init__()

        transforms = [Augmentation.transforms[k] for k in params.keys()]
        transform_params = [params[k] for k in params.keys()]

        # add identity parameters for all transforms and convert to list
        transform_params = [t.prepare(params) for t, params in zip(transforms, transform_params)]

        # get all combinations of transforms params
        transform_params = list(itertools.product(*transform_params))

        self.forward_aug = [t.forward for t in transforms]
        self.forward_params = transform_params

        self.backward_aug = [t.backward for t in transforms[::-1]] # reverse transforms
        self.backward_params = [p[::-1] for p in transform_params] # reverse params

        self.n_transforms = len(transform_params)

    @property
    def forward(self):
        return self.forward_aug, self.forward_params

    @property
    def backward(self):
        return self.backward_aug, self.backward_params
