"""Implementations of permutation-like transforms."""

import torch
import numpy as np
import glog as logger

from manifold import transforms
from manifold import various


class Permutation(transforms.Transform):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, device, permutation, dim=1):
        if permutation.ndimension() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not various.is_positive_int(dim):
            raise ValueError("dim must be a positive integer.")

        super().__init__()
        self.device = device
        self._dim = dim
        self.register_buffer("_permutation", permutation)

    @property
    def _inverse_permutation(self):
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(device, inputs, permutation, dim, full_jacobian=False):
        if dim >= inputs.ndimension():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError("Dimension {} in inputs must be of size {}.".format(dim, len(permutation)))
        batch_size = inputs.shape[0]
        if full_jacobian:
            outputs = torch.index_select(inputs, dim, permutation)

            # The brute force way does not seem to work, not sure why, maybe index_select breaks autodiff
            # jacobian = utils.batch_jacobian(outputs, inputs)

            # First build the Jacobian as a 2D matrix
            jacobian = torch.zeros((outputs.size()[dim], inputs.size()[dim])).to(device)
            jacobian[permutation, torch.arange(0, len(permutation), 1)] = 1.0

            # Add dummy dimensions for batch size...
            jacobian = jacobian.unsqueeze(0)  # (1, n, n)
            # ... and for every dimension smaller than dim...
            for i in range(dim - 1):
                jacobian = jacobian.unsqueeze(2 + 2 * i)
                jacobian = jacobian.unsqueeze(1 + i)
            # ... and for every dimension larger than dim...
            for i in range(len(inputs.size()) - dim - 1):
                jacobian = jacobian.unsqueeze(1 + 2 * dim + 2 * i)
                jacobian = jacobian.unsqueeze(1 + dim + i)

            # Broadcast to full size
            jacobian = torch.ones(outputs.size() + inputs.size()[1:]).to(device) * jacobian

            # Finally, view it as a (batch, n, n) Jacobian
            jacobian = jacobian.view((inputs.size()[0], np.prod(inputs.size()[1:]), np.prod(inputs.size()[1:])))

            return outputs, jacobian
        else:
            outputs = torch.index_select(inputs, dim, permutation)
            logabsdet = torch.zeros(batch_size).to(device)
            return outputs, logabsdet

    def forward(self, inputs, context=None, full_jacobian=False):
        return self._permute(self.device, inputs, self._permutation, self._dim, full_jacobian=full_jacobian)

    def inverse(self, inputs, context=None, full_jacobian=False):
        return self._permute(self.device, inputs, self._inverse_permutation, self._dim, full_jacobian=full_jacobian)


class RandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, device, features, dim=1):
        if not various.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(device, torch.randperm(features), dim)

class MaskBasedPermutation(Permutation):
    """ Given a 1D binary mask, permutes inputs such that active elements come first, followed by inactive elements """

    def __init__(self, device, mask, dim=1):
        idx = torch.cat((torch.nonzero(mask).squeeze(), torch.nonzero(1 - mask).squeeze()), 0)

        super().__init__(device=device, permutation=idx, dim=dim)
        