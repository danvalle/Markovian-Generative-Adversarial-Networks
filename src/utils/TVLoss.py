import torch
import torch.nn as nn
from torch.autograd import Function


class TVLoss(nn.Module):
    
    def __init__(self, TVLoss_weight=1e-4):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = torch.Tensor(1)
        self.TVLoss_weight.fill_(TVLoss_weight)

    def forward(self, x):
        tv_loss = tv_function.apply
        x = tv_loss(x, self.TVLoss_weight)
        return x


class tv_function(Function):

    @staticmethod
    def forward(ctx, input, strength):
        ctx.save_for_backward(input, strength)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, strength = ctx.saved_tensors

        C, H, W = input.size()[1:]
        x_diff = input[:, :, :-1, :-1].clone()
        x_diff.sub_(input[:, :, :-1, 1:])
        y_diff = input[:, :, :-1, :-1].clone()
        y_diff.sub_(input[:, :, 1:, :-1])

        grad_input = torch.zeros_like(input)
        grad_input[:, :, :-1, :-1].add_(x_diff).add_(y_diff)
        grad_input[:, :, :-1, 1:].sub_(x_diff)
        grad_input[:, :, 1:, :-1].sub_(y_diff)

        grad_input.mul_(strength.item())
        grad_input.add_(grad_output)
        return grad_input, None
