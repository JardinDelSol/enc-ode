import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from inspect import signature
import math
import torch.nn.functional as F
import time

# Adapted from rtqichen/ffjord
# https://github.com/rtqichen/ffjord

class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)

        self._age_enc = nn.Linear(1, 16)

    def forward(self, t, x, e):
        x = torch.cat((x, e), -1)
        result = self._layer(x)
        return result


class ActNorm(nn.Module):
    def __init__(self, num_features, init_scale=1.0):
        super(ActNorm, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.init_scale = init_scale
        self.register_buffer("initialized", torch.tensor(0))

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_ = x.reshape(-1, x.shape[-1])
                batch_mean = torch.mean(x_, dim=0)
                batch_var = torch.var(x_, dim=0)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(
                    -0.5 * torch.log(batch_var) + math.log(self.init_scale)
                )
                self.initialized.fill_(1)

        bias = self.bias.expand_as(x)
        weight = self.weight.expand_as(x)

        y = (x + bias) * F.softplus(weight)
        return y

    def __repr__(self):
        return "{name}({num_features})".format(
            name=self.__class__.__name__, **self.__dict__
        )


class SequentialDiffEq(nn.Module):
    """A container for a sequential chain of layers. Supports both regular and diffeq layers."""

    def __init__(self, *layers):
        super(SequentialDiffEq, self).__init__()
        self.layers = nn.ModuleList([diffeq_wrapper(layer) for layer in layers])

    def forward(self, t, x, e):
        for layer in self.layers:
            x = layer(t, x, e)
        return x


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)


class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super(DiffEqWrapper, self).__init__()
        self.module = module

    def forward(self, t, y, e):
        if len(signature(self.module.forward).parameters) == 1:
            return self.module(y)
        elif len(signature(self.module.forward).parameters) == 3:
            return self.module(t, y, e)
        else:
            raise ValueError(
                "Differential equation needs to either take (t, y) or (y,) as input."
            )

    def __repr__(self):
        return self.module.__repr__()


class IntensityODEFunc(nn.Module):
    def __init__(self, hdim, dstate_fn):
        super().__init__()
        self.hdim = hdim
        self.dstate_fn = dstate_fn

    def forward(self, t, state):
        (f_hat, l) = state
        d_ht = self.dstate_fn(t, f_hat, l)
        return (d_ht, torch.zeros_like(l))


class TimeVariableODE(nn.Module):
    start_time = 0.0
    end_time = 1.0
    # end_time = 1.0e2

    def __init__(
        self, func, atol=1e-6, rtol=1e-6, method="dopri5", energy_regularization=0.01
    ):
        super().__init__()
        self.func = func
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.energy_regularization = energy_regularization
        self.nfe = 0
        self.testing = False

    def integrate(self, t0, t1, x0, nlinspace=1, method=None):
        assert nlinspace > 0  # number of sections
        method = method or self.method
        self.nfe = 0

        solution = odeint(  # odeint_adjoint
            self,  # self as a function
            (
                t0,
                t1,
                torch.zeros(1).to(x0[0]),
                *x0,
            ),
            torch.linspace(self.start_time, self.end_time, nlinspace + 1).to(t0),
            method=method,
        )
        _, _, energy, *xs = solution  # xs[0][0] == x0[0]
        reg = energy * self.energy_regularization
        # return WrapRegularization.apply(reg, *xs)  # Q: 1 for reg
        return xs

    def forward(self, s, state):
        """Solves the same dynamics but uses a dummy variable that always integrates [0, 1]."""
        self.nfe += 1
        t0, t1, e, *x = state

        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0

        with torch.enable_grad():
            x = tuple(x_.requires_grad_(True) for x_ in x)
            dx = self.func(t, x)

            dx = tuple(dx_ * ratio.reshape(-1, *([1] * (dx_.ndim - 1))) for dx_ in dx)

            d_energy = sum(torch.sum(dx_ * dx_) for dx_ in dx) / sum(
                x_.numel() for x_ in x
            )

        dx = tuple(dx_.detach() for dx_ in dx)

        result = tuple(
            [
                torch.zeros_like(t0),
                torch.zeros_like(t1),
                d_energy,
                *dx,
            ]
        )
        return result


class WrapRegularization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reg, *x):
        ctx.save_for_backward(reg)
        return x

    @staticmethod
    def backward(ctx, *grad_x):
        (reg,) = ctx.saved_variables
        return (torch.ones_like(reg), *grad_x)
