"""Implementations of Normal distributions."""

import numpy as np
import torch

from manifold.distributions import Distribution
from manifold import various


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)
        self._log_z = 0.5 * np.prod(shape) * np.log(2 * np.pi)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError("Expected input of shape {}, got {}".format(self._shape, inputs.shape[1:]))
        neg_energy = -0.5 * various.sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return torch.randn(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape)
            return various.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)

class RescaledNormal(Distribution):
    """A multivariate Normal with zero mean and a diagonal covariance that is epsilon^2 along each diagonal entry of the matrix."""

    def __init__(self, shape, std=1.0, clip=10.0):
        super().__init__()
        self._shape = torch.Size(shape)
        self.std = std
        self._clip = clip
        self._log_z = 0.5 * np.prod(shape) * np.log(2 * np.pi) + np.prod(shape) * np.log(self.std)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError("Expected input of shape {}, got {}".format(self._shape, inputs.shape[1:]))
        if self._clip is not None:
            inputs = torch.clamp(inputs, -self._clip, self._clip)
        neg_energy = -0.5 * various.sum_except_batch(inputs ** 2, num_batch_dims=1) / self.std ** 2
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return self.std * torch.randn(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]
            samples = self.std * torch.randn(context_size * num_samples, *self._shape)
            return various.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)
