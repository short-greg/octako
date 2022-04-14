from abc import ABC, abstractmethod, abstractproperty
import typing
import torch as th
import torch.nn as nn
from functools import partial, singledispatch
import uuid


UNDEFINED = object()

# Test what i have... then continue to build upon it


class Node(ABC):
    
    @abstractproperty
    def y(self):
        raise NotImplementedError

    @y.setter
    def y(self, y):
        raise NotImplementedError

    @abstractproperty
    def x(self):
        raise NotImplementedError

    @x.setter
    def x(self, x):
        raise NotImplementedError
    
    def to(self):
        raise NotImplementedError


class F(nn.Module):

    def __init__(self, f: typing.Callable, *args, **kwargs):
        self._f = partial(f, *args, **kwargs)
    
    def forward(self, *x):
        return self._f(*x)


class Layer(Node):

    def __init__(
        self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
        x=UNDEFINED, incoming: Node=None
    ):
        if isinstance(nn_module, list):
            nn_module = nn.Sequential(nn_module)
        self._module = nn_module
        self._x = x
        self._y = UNDEFINED
        self._incoming = incoming if self._x is not UNDEFINED else None

    @property
    def y(self):
        if self._y == UNDEFINED and self._x != UNDEFINED:
            self._y = self._module(self._x)
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x
    
    def to(self, nn_module: typing.Union[typing.List[nn.Module], nn.Module]):
        return Layer(nn_module, x=self._y, incoming=self)
    
    def probe(self):
        pass

    # will assume the output
    def __getitem__(self, idx: int):
        pass


# want two types of "in" nodes
# one that has a module and no x and
# one that has an x and no module
# figure out how to organize these

class Root(Node):

    def __init__(self, nn_module: typing.Union[typing.List[nn.Module], nn.Module]):

        if isinstance(nn_module, list):
            nn_module = nn.Sequential(nn_module)
        self._nn_module: nn.Module = nn_module
        self._y = UNDEFINED
    
    @property
    def x(self):
        return

    @x.setter
    def x(self, x):
        raise RuntimeError('Cannot set the input to a root node')

    @property
    def y(self):
        if self._y is UNDEFINED:
            self._y = self._nn_module()
        return self._y

    @y.setter
    def y(self, y):
        self._y = y
    
    def reset(self):
        self._y = UNDEFINED


class In(Node):

    def __init__(self, x=UNDEFINED):
        # may want to allow x to be a module
        self._x = x

    @property
    def y(self):
        return self._x

    @y.setter
    def y(self, y):
        self._x = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    def to(self, nn_module: typing.Union[typing.List[nn.Module], nn.Module]):
        return Layer(nn_module, x=self._x, incoming=self)

    def probe(self):
        pass


# probe({id: value})


# join = layer.join(layer2)
class Join(Node):
    pass

# route
# route = layer.route()
# with route.if_() as if_:
# with route.else_() as else_:

class Route(Node):
    pass


class Index(Node):
    pass


class Tako(nn.Module):

    @abstractmethod
    def forward_iter(self, in_: In, deep: bool=False) -> typing.Iterator:
        pass



