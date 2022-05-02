from typing import Any
import typing
import torch as th
from functools import partial
import torch.nn as nn
from .utils import UNDEFINED


class Null(nn.Module):
    """
    Module that does not affect the input
    """

    def __init__(self, multiple_xs: bool=False):
        """initializer

        Args:
            multi (bool, optional): Whether there are multiple outputs. Defaults to False.
        """
        super().__init__()
        self.multiple_xs = multiple_xs

    def forward(self, *x):
        if not self.multiple_xs:
            return x[0]
        return x

class Shared(object):

    def __init__(self, obj: object, member: str):

        self.obj = obj
        self.member = member

    def __call__(self):
        return getattr(self.obj, self.member)


class Set(nn.Module):

    def __init__(self, nn_module: nn.Module, member: str, val: Any):

        super().__init__()
        self.module = nn_module
        self.val = val
        self.member = member

    def forward(self, x):

        val = self.val() if isinstance(self.val, Shared) else self.val
        setattr(self.module, self.member, val)
        return self.module(x)


class Case(nn.Module):

    def __init__(self, condition: typing.Callable, module: nn.Module):
        super().__init__()
        self.condition = condition
        self.module = module
    
    def forward(self, case, x):
        if self.condition(case):
            return self.module(x)
        return x


class F(nn.Module):

    def __init__(self, f: typing.Callable, *args, **kwargs):
        super().__init__()
        self._f = f
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, x):
        return self._f(x, *self.args, **self.kwargs)


class Gen(nn.Module):

    def __init__(self, generator: typing.Callable[[], th.Tensor], *args, **kwargs):
        super().__init__()
        self._f = generator
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, x: bool):

        if x is True:
            return self._f(*self.args, **self.kwargs)
        return UNDEFINED


# class Lambda(nn.Module):
#     """
#     Define a module inline
#     """

#     def __init__(
#         self, lambda_fn: typing.Callable[[], th.Tensor], *args, **kwargs
#     ):
#         """initializer

#         Args:
#             lambda_fn (typing.Callable[[], torch.Tensor]): Function to process the tensor
#         """

#         super().__init__()
#         self._lambda_fn = lambda_fn
#         self._args = args
#         self._kwargs = kwargs
    
#     def forward(self, *x: th.Tensor):
#         """Execute the lambda function

#         Returns:
#             list[torch.Tensor] or torch.Tensor 
#         """
#         return self._lambda_fn(*x, *self._args, **self._kwargs)

