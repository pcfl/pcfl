import numpy as np
import torch

from manifold import transforms
from manifold import various

class LogTanh(transforms.Transform):
    """Tanh with unbounded output. Constructed by selecting a cut_point, and replacing values to
    the right of cut_point with alpha * log(beta * x), and to the left of -cut_point with
    -alpha * log(-beta * x). alpha and beta are set to match the value and the first derivative of
    tanh at cut_point."""

    def __init__(self, cut_point=1):
        if cut_point <= 0:
            raise ValueError("Cut point must be positive.")
        super().__init__()

        self.cut_point = cut_point
        self.inv_cut_point = np.tanh(cut_point)

        self.alpha = (1 - np.tanh(np.tanh(cut_point))) / cut_point
        self.beta = np.exp((np.tanh(cut_point) - self.alpha * np.log(cut_point)) / self.alpha)

    def forward(self, inputs, context=None, full_jacobian=False):
        if full_jacobian:
            raise NotImplementedError

        mask_right = inputs > self.cut_point
        mask_left = inputs < -self.cut_point
        mask_middle = ~(mask_right | mask_left)

        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = torch.tanh(inputs[mask_middle])
        outputs[mask_right] = self.alpha * torch.log(self.beta * inputs[mask_right])
        outputs[mask_left] = self.alpha * -torch.log(-self.beta * inputs[mask_left])

        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = torch.log(1 - outputs[mask_middle] ** 2)
        logabsdet[mask_right] = torch.log(self.alpha / inputs[mask_right])
        logabsdet[mask_left] = torch.log(-self.alpha / inputs[mask_left])
        logabsdet = various.sum_except_batch(logabsdet, num_batch_dims=1)

        return outputs, logabsdet

    def inverse(self, inputs, context=None, full_jacobian=False):
        if full_jacobian:
            raise NotImplementedError

        mask_right = inputs > self.inv_cut_point
        mask_left = inputs < -self.inv_cut_point
        mask_middle = ~(mask_right | mask_left)

        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = 0.5 * torch.log((1 + inputs[mask_middle]) / (1 - inputs[mask_middle]))
        outputs[mask_right] = torch.exp(inputs[mask_right] / self.alpha) / self.beta
        outputs[mask_left] = -torch.exp(-inputs[mask_left] / self.alpha) / self.beta

        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = -torch.log(1 - inputs[mask_middle] ** 2)
        logabsdet[mask_right] = -np.log(self.alpha * self.beta) + inputs[mask_right] / self.alpha
        logabsdet[mask_left] = -np.log(self.alpha * self.beta) - inputs[mask_left] / self.alpha
        logabsdet = various.sum_except_batch(logabsdet, num_batch_dims=1)

        return outputs, logabsdet