from typing import Callable

import torch
from torch import autograd, nn
import torch.nn.functional as F


class FractionalDecoupling(autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.round() # f(x) = x


    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryDecoupling(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class StraighThroughReLU(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class StraightThroughEstimator(nn.Module):
    def __init__(self, function: Callable = FractionalDecoupling.apply):
        super(StraightThroughEstimator, self).__init__()
        self.function = function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.function(x)
        return x