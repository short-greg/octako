from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from mailbox import NotEmptyError
import typing
import torch as th
import torch.nn as nn
from functools import partial, singledispatch
import uuid
import time


UNDEFINED = object()
EMPTY = object()
# EMPTY <- determine how to handle this....


# add incoming layer


# Node = typing.TypeVar('Node')
# Layer = typing.TypeVar('Layer')
# Join = typing.TypeVar('Join')


# Limit to these two types of nodes
# -> Join needs to be a module
# -> Route needs to be a module
# etc
# In
# Layer

class ID(object):

    def __init__(self, id: uuid.UUID=None):

        self.x = id if id is not None else uuid.uuid4()


@dataclass
class Info:
    name: str = None
    id: str = None
    tags: typing.List[str] = None
    annotation: str = None


def if_true(val, obj):
    if val is True:
        return obj


class Node(ABC):

    def __init__(
        self, is_outgoing: bool=False, incoming=None, info: Info=None
    ):
        self._info = info or Info()
        self.is_outgoing = is_outgoing
        self._incoming = incoming

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
    
    def to(
        self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
        info: Info=None
    ):
        incoming = if_true(self.is_outgoing, self)
        return Layer(
            nn_module, x=self.y, incoming=incoming, 
            info=info, is_outgoing=self.is_outgoing
        )

    def empty(
        self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
        info: Info=None
    ):
        incoming = if_true(self.is_outgoing, self)
        return Layer(
            nn_module, x=EMPTY, incoming=incoming, 
            info=info, is_outgoing=self.is_outgoing
        )

    @abstractproperty
    def is_parent(self):
        raise NotImplementedError

    def sub(self) -> typing.Iterator:
        if self.is_parent:
            for layer in self._module.forward_iter(In(self._x)):
                yield layer

    def join(self, *others, info: Info=None):
        outgoing = self.is_outgoing
        is_undefined = self.y == UNDEFINED
        other_ys = []
        for other in others:
            outgoing = outgoing or other.is_outgoing
            is_undefined = is_undefined or other.y == UNDEFINED
            other_ys.append(other.y)
        incoming = if_true(outgoing, self)

        # TODO: think about this more
        # y = UNDEFINED if undefined else other_ys
        if is_undefined: x = UNDEFINED
        else: x = [self.y, *other_ys]
        
        return Layer(
            nn_module=Join(len(other_ys) + 1),
            x=x, 
            incoming=incoming, is_outgoing=outgoing, info=info
        )

    def route(self):
        pass

    def get(self, idx: typing.Union[slice, int], info: Info=None):
        incoming = if_true(self.is_outgoing, self)
        return Layer(
            Index(idx), x=self.y, incoming=incoming, 
            is_outgoing=self.is_outgoing, info=info
        )

    def __getitem__(self, idx: int):
        return self.get(idx)


class Layer(Node):

    def __init__(
        self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
        x=UNDEFINED, incoming: typing.Optional[Node]=None, info: Info=None, 
        is_outgoing: bool=False
    ):
        super().__init__(is_outgoing, incoming, info=info)
        if isinstance(nn_module, typing.List):
            nn_module = nn.Sequential(*nn_module)
        self._module = nn_module
        self._x = x
        self._y = UNDEFINED
        self._incoming = incoming

    @property
    def y(self):
        if self._y == UNDEFINED and self._x != UNDEFINED:
            self._y = self._module(self.x) if self._x != EMPTY else self._module()
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

    @property
    def is_parent(self):
        return isinstance(self._module, Tako)

    def sub(self) -> typing.Iterator:
        if self.is_parent:
            for layer in self._module.forward_iter(In(self._x)):
                yield layer


# want two types of "in" nodes
# one that has a module and no x and
# one that has an x and no module
# figure out how to organize these


class In(Node):

    def __init__(self, x=UNDEFINED, is_outgoing: bool=False, info: Info=None):
        super().__init__(is_outgoing, None, info)
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

    @property
    def is_parent(self):
        return False


class Join(nn.Module):
    
    def __init__(self, n_modules: int):
        super().__init__()
        self._n_modules = n_modules

    def forward(self, x):
        if len(x) > self._n_modules:
            raise RuntimeError(f"Number of inputs must be equal to {self._n_modules}")
        return x


class Index(nn.Module):
    
    def __init__(self, idx: typing.Union[int, slice]):
        super().__init__()
        self._idx = idx

    def forward(self, x):
        return x[self._idx]



class Tako(nn.Module):

    @abstractmethod
    def forward_iter(self, in_: Node) -> typing.Iterator:
        pass

    def forward(self, x):
        y = x
        for layer in self.forward_iter(In(x)):
            y = layer.y
        return y


class Sequence(Tako):

    def __init__(self, modules):
        super().__init__()
        self._modules = modules

    def forward_iter(self, in_: Node):

        cur = in_ or In()
        for module in self._modules:
            cur = cur.to(module)
            yield cur


class Process(ABC):

    @abstractmethod
    def apply(self, node: Node):
        pass


class LambdaProcess(Process):

    def __init__(self, f):
        self._f = f

    def apply(self, node: Node):
        self._f(node)


class NodeSet(object):

    def __init__(self, nodes: typing.List[Node]):
        self._nodes = nodes
    
    def apply(self, process: Process):
        for node in self._nodes:
            process.apply(node)


class Filter(ABC):

    @abstractmethod
    def check(self, layer: Layer) -> bool:
        pass

    def extract(self, tako: Tako):
        return NodeSet(
            [layer for layer in self.filter(tako) if self.check(layer)]
        )
    
    def apply(self, tako: Tako, process: Process):
        for layer in self.filter(tako):
            if self.check(layer):
                process.apply(layer)

    @abstractmethod
    def filter(self, tako) -> typing.Iterator:
        pass

# filter.apply(tako, lambda layer: layer.set('a', 1))

def layer_dive(layer: Layer):
    """Loop over all sub layers in a layer including 

    Args:
        layer (Layer): _description_

    Yields:
        _type_: _description_
    """
    for layer_i in layer.sub():
        if layer_i.is_parent:
            for layer_j in layer_dive(layer_i):
                yield layer_j
        else: yield layer_i


def dive(tako: Tako, in_):
    for layer in tako.forward_iter(in_):
        yield layer_dive(layer)


# route
# route = layer.route()
# with route.if_() as if_:
# with route.else_() as else_:

class If_(object):

    def __enter__(self):
        pass


class Route(Node):
    
    def __init__(self, x=UNDEFINED, incoming: Node=None, is_outgoing: bool=False, info: Info=None):
        pass

    def if_(self, node: Node):

        pass


class Loop(Node):
    pass



# class Index(Node):
    
#     def __init__(self, idx: typing.Union[int, slice], x=UNDEFINED, incoming: Node=None, is_outgoing: bool=False, info: Info=None):
#         super().__init__(is_outgoing, incoming, info)
#         self._idx = idx
#         self._x = x

#     @property
#     def y(self):
#         if self._y is not UNDEFINED:
#             return self._y
#         if self._x is UNDEFINED:
#             return UNDEFINED

#         return self._x[self._idx]

#     @y.setter
#     def y(self, y):
#         self._y = y

#     @property
#     def x(self):
#         return self._x

#     @x.setter
#     def x(self, x):
#         self._x = x

#     def to(self, nn_module: typing.Union[typing.List[nn.Module], nn.Module]):
#         return Layer(nn_module, x=self.y, incoming=self)


# class Join(Node):
    
#     def __init__(self, x=UNDEFINED, incoming: Node=None, is_outgoing: bool=False, info: Info=None):
#         super().__init__(is_outgoing, Info)
#         self._x = x
#         self._incoming = incoming

#     @property
#     def y(self):
#         if self._y is not UNDEFINED:
#             return self._y

#         return self._x

#     @y.setter
#     def y(self, y):
#         self._y = y

#     @property
#     def x(self):
#         return self._x

#     @x.setter
#     def x(self, x):
#         self._x = x

#     def to(self, nn_module: typing.Union[typing.List[nn.Module], nn.Module]):
#         return Layer(nn_module, x=self.y, incoming=self)

