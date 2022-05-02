from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from operator import is_
from re import X
import typing
from numpy import choose, isin
import torch as th
import torch.nn as nn
from functools import partial, singledispatch, singledispatchmethod
import uuid
import time
from .utils import  ID, UNDEFINED
from .modules import F


def first(x):

    for x_i in x:
        if x_i != UNDEFINED:
            return x_i
    return UNDEFINED


def require(x):

    for x_i in x:
        if x_i == UNDEFINED:
            return UNDEFINED
    return x


@dataclass
class Info:
    name: str = None
    id: str = None
    tags: typing.List[str] = None
    annotation: str = None


class Incoming(object):

    def __init__(self, node):   

        self._is_value = True

        if node.y is UNDEFINED:
            self._is_value = False
            self._in = node
        else:
            self._in = node.y

    @property
    def x(self):
        if self._is_value:
            return self._in
        return UNDEFINED

    def probe(self, by):
        if self._is_value:
            return self._in
        return self._in.probe(by)


class Feedback(object):

    # use to get function type in feedback
    def _f():
        pass

    def __init__(self, key, delay: int=0):

        self._key = key
        self._delay = delay
        self._responses = [None] * (delay + 2)
        self._set = [False] * (delay + 2)
        self._delay_i = delay
        self._cur_i = 0
    
    def adv(self):
        self._responses = self._responses[1:] + [None]
        self._set = self._set[1:] + [False]
        self._delay_i = max(self._delay_i - 1, 0)
        self._cur_i += 1

    def set(self, value, adv=False):
        self._responses[self._delay + 1] = value
        self._set[self._delay + 1] = True
        if adv:
            self.adv()

    def get(self, default=None):
        
        if not self._set[0]:
            if default is None: return None
            return default() # if isinstance(default, type(self._f)) else default
        return self._responses[0]


def is_defined(x):
    return not isinstance(x, Incoming) and x != UNDEFINED


def to_incoming(node):

    if node.y is UNDEFINED:
        return Incoming(node)
    return node.y


class Node(ABC):

    def __init__(
        self, x=UNDEFINED, info: Info=None
    ):
        self._info = info or Info()
        self._x = x

    @abstractproperty
    def y(self):
        raise NotImplementedError

    @y.setter
    def y(self, y):
        raise NotImplementedError

    @property
    def x(self):
        if isinstance(self._x, Incoming): return UNDEFINED
        return self._x

    @x.setter
    def x(self, x):
        self._x = x
    
    def to(
        self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
        info: Info=None
    ):
        return Layer(
            nn_module, x=to_incoming(self), info=info
        )

    def check_id(self, id: ID):
        if self._info.id is None:
            return False
        return self._info.id == id

    def join(self, *others, info: Info=None, join_f=require):
        
        ys = []
        
        for node in [self, *others]:
            ys.append(to_incoming(node))

        return Join(
            x=ys, info=info
        )

    def route(self, cond_node):

        return Decision(
            to_incoming(cond_node), to_incoming(self)
        ), Decision(
            to_incoming(cond_node), to_incoming(self), positive=False
        )

    def get(self, idx: typing.Union[slice, int], info: Info=None):
        return Index(
            idx, x=to_incoming(self), info=info
        )
    
    def loop(self, f, *filters, info: Info=None):

        loop = Loop(x=to_incoming(self), info=info)
        
        accumulators = []
        for filters in zip(filters):
            accumulators.append(loop.filter(filter))
        while True:
            layers = f(loop)
            for layer, accumulator in zip(layers, accumulators):
                accumulator.from_(layer)
            loop.adv()
            if loop.is_end():
                break
        return accumulators

    def __getitem__(self, idx: int):
        return self.get(idx)


class Cur(nn.Module):

    def __init__(self, default_f):
        super().__init__()
        self._default_f = default_f

    def forward(self, x: Feedback):
        return x.get(self._default_f)


def cur(feedback: Node, default_f):
    return feedback.to(Cur(default_f))


class Process(ABC):

    @abstractmethod
    def apply(self, node: Node):
        pass


class NodeSet(object):

    def __init__(self, nodes: typing.List[Node]):
        self._nodes = nodes
    
    def apply(self, process: Process):
        for node in self._nodes:
            process.apply(node)


class LambdaProcess(Process):

    def __init__(self, f):
        self._f = f

    def apply(self, node: Node):
        self._f(node)


class Decision(Node):

    def __init__(
        self, cond_x=UNDEFINED, x=UNDEFINED, positive: bool=True, info: Info=None
    ):
        super().__init__(x, info=info)
        self._y = UNDEFINED
        self._cond_x = cond_x
        self._positive = positive

    @property
    def y(self):

        if not is_defined(self._cond_x) or self._cond_x is not self._positive:
            return UNDEFINED
        elif not is_defined(self._x):
            return UNDEFINED
        return self._x

    @y.setter
    def y(self, y):
        self._y = y


class Join(Node):

    def __init__(
        self, x=UNDEFINED, info: Info=None
    ):
        super().__init__(x, info=info)
        self._y = UNDEFINED

    @property
    def y(self):

        undefined = False
        for x_i in self._x:
            if isinstance(x_i, Incoming):
                undefined = True
        
        if undefined:
            self._y = UNDEFINED
        else:
            self._y = self._x

        return self._y

    @y.setter
    def y(self, y):
        self._y = y


class Index(Node):
    
    def __init__(self, idx: typing.Union[int, slice], x=UNDEFINED, info: Info=None):
        super().__init__(x, info)
        self._idx = idx

    @property
    def y(self):
        if isinstance(self._x, Incoming):
            return UNDEFINED
        else:
            return self._x[self._idx]


class Layer(Node):

    def __init__(
        self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
        x=UNDEFINED, info: Info=None
    ):
        super().__init__(x, info=info)
        if isinstance(nn_module, typing.List):
            nn_module = nn.Sequential(*nn_module)
        self.op = nn_module
        self._y = UNDEFINED

    @property
    def y(self):

        if self._y == UNDEFINED and isinstance(self._x, Incoming):
            return UNDEFINED

        elif self._y == UNDEFINED and self._x != UNDEFINED:
            self._y = self.op(self._x)

        return self._y
    
    @y.setter
    def y(self, y):
        self._y = y


class In(Node):
    
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._x

    @y.setter
    def y(self, y):
        self._x = y

    @property
    def is_parent(self):
        return False


class Loop(Node):

    def adv(self) -> bool:
        if not self.is_end():
            self._x.adv()
            return True

        return False

    def to(self, nn_module, info):
        if self.y is UNDEFINED:
            return Layer(
                nn_module, x=Incoming(self), info=info
            )
        return Layer(
            nn_module, x=self.y, info=info
        )
    
    def filter(self, filter, info: Info=None):
        return Accumulator(
            self, filter, info=info
        )

    @property
    def y(self):
        if self._x is UNDEFINED or self._x.is_end():
            self._y = UNDEFINED
        else:
            self._y = self._x.cur
        return self._y
        
    def is_end(self):
        
        return self._x is UNDEFINED or self._x.is_end()


class Filter(nn.Module):

    @abstractmethod
    def update(self, x, state=None):
        pass

    @abstractmethod
    def forward(self, x, state=None):
        pass


class All(Filter):

    def update(self, x, state=None):
        if state is None:
            state = [x, *state]
        return state
    
    def forward(self, state):
        return th.stack(state)


class Last(Filter):

    def update(self, x, state=None): 
        return x
    
    def forward(self, state):
        return state


class Accumulator(Node):

    def __init__(self, loop: Loop, filter: Filter, x=UNDEFINED, info=None):
        super().__init__(x, info)
        self._loop = loop
        self._filter: Filter = filter
        self._incoming = None
        self._state = None
        
    def from_(self, node: Node):
        self._incoming = node
        if node.y is not UNDEFINED:
            self._state = self._filter.update(node.y, self._state)

    @property
    def y(self):
        
        if self._state is not None:
            self._y = self._filter(self._state)
        else:
            self._y = UNDEFINED
        return self._y

    # probe needs to check if the loop is at the end
    # and advance the loop.. if needed
    def probe(self):
        pass


class Iterator(ABC):

    @abstractmethod
    def adv(self) -> bool:
        pass

    @abstractmethod
    def is_end(self) -> bool:
        pass

    @abstractproperty
    def cur(self):
        pass


class Iterate(nn.Module):

    @abstractmethod
    def forward(self, x) -> Iterator:
        pass


# class Failure(nn.Module):
    
#     def forward(self, x):

#         if x[0] is False:
#             return x[1]
#         return UNDEFINED


# class Success(nn.Module):
    
#     def forward(self, x):

#         if x[0] is True:
#             return x[1]
#         return UNDEFINED


# I think i can do it like this
#
# iterator = x.to(Iterator())
# accumulator = iterator.accumulate()

# with iterator.iterate():
#     yield iterator.cur

#     layer = iterator.cur.to(nn.Linear())
#     yield layer
#     accumulator.from_(layer)

# yield accumulator

# class Iterator(object):

#     @abstractmethod
#     def iterator(self):
#         raise NotImplementedError
    
#     @abstractmethod
#     def adv(self) -> bool:
#         raise NotImplementedError
    
#     @abstractmethod
#     def is_end(self) -> bool:
#         raise NotImplementedError
    
#     @abstractmethod
#     def reset(self):
#         raise NotImplementedError
    
#     @abstractmethod
#     def respond(self, y):
#         raise NotImplementedError


# class ListIterator(Iterator):

#     def __init__(self, l: list):
#         self._list = l
#         self._idx = 0
    
#     def adv(self) -> bool:
#         if self._idx < len(self._list):
#             self._idx += 1
#             return True
#         return False
    
#     def is_end(self) -> bool:
#         return self._idx == len(self._list) - 1
    
#     def cur(self):
#         if self.is_end(): return UNDEFINED
#         return self._list[self._idx]
    
#     def response(self, y):
#         pass

# add in¥oming layer


# Node = typing.TypeVar('Node')
# Layer = typing.TypeVar('Layer')
# Join = typing.TypeVar('Join')


# Limit to these two types of nodes
# -> Join needs to be a module
# -> Route needs to be a module
# etc
# In
# Layer