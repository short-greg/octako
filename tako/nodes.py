from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
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

    def __init__(self, incoming=[]):

        self._is_list = isinstance(incoming, list)
        self._incoming = incoming
        self._defined = incoming is None

    def add(self, node):

        self._incoming.append(node)


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


class Node(ABC):

    def __init__(
        self, x, info: Info=None
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
        return self._x

    @x.setter
    def x(self, x):
        self._x = x
    
    def to(
        self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
        info: Info=None
    ):
        return Layer(
            nn_module, x=self.y, info=info
        )

    @abstractproperty
    def is_parent(self):
        raise NotImplementedError

    def check_id(self, id: ID):
        if self._info.id is None:
            return False
        return self._info.id == id

    def children(self) -> typing.Iterator:
        if self.is_parent:
            for layer in self._module.forward_iter(In(self._x)):
                yield layer

    def join(self, *others, info: Info=None, join_f=require):
        
        ys = []
        defined = True
        for node in [self, *others]:
            if not is_defined(node.y):
                defined = False
            ys.append(node.y)
        
        if not defined:
            ys = Incoming(ys)
        return Layer(
            nn_module=F(join_f),
            x=ys, info=info
        )

    def route(self, cond_node):

        ys = [cond_node.y, self.y]
        if not (is_defined(cond_node.y) and is_defined(self.y)):
            ys = Incoming(ys)

        return Layer(
            Success(), x=ys
        ), Layer(
            Failure(), x=ys
        )

    def get(self, idx: typing.Union[slice, int], info: Info=None):
        return Layer(
            Index(idx), x=self.y, info=info
        )
    
    def iterate(self, f, n_accumulators: int=1):

        iterator = self.to(Iterator(x=self.y))
        accumulators = []
        for _ in range(n_accumulators):
            accumulators.append(iterator.accumulate())
        while True:
            f(iterator.cur, *accumulators)
            iterator.adv()
            if iterator.is_end():
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
            return Incoming(self)

        elif self._y == UNDEFINED and self._x != UNDEFINED:

            # # Think whether to keep this
            # if isinstance(self._x, Iterator):
            #     self._y = self._eval_iterator(self._x)
            # else:
            self._y = self._eval(self._x)

        return self._y
    
    # def _eval_iterator(self, x):
    #     x: Iterator = x
    #     y = []
    #     while True:
    #         if x.cur == UNDEFINED:
    #             break
    #         y.append(self._eval(x.cur))
    #         x.adv()
    #         x.respond(y[-1])
    #     return y

    def _eval(self, x):
        return self.op(x)

    @y.setter
    def y(self, y):
        self._y = y


class Failure(nn.Module):
    
    def forward(self, x):

        if x[0] is False:
            return x[1]
        return UNDEFINED


class Success(nn.Module):
    
    def forward(self, x):

        if x[0] is True:
            return x[1]
        return UNDEFINED


class In(Node):

    def __init__(self, x=UNDEFINED, info: Info=None):
        super().__init__(x, info)

    @property
    def y(self):
        if self._x == UNDEFINED:
            return Incoming(self)
        return self._x

    @y.setter
    def y(self, y):
        self._x = y

    @property
    def is_parent(self):
        return False


class Index(nn.Module):
    
    def __init__(self, idx: typing.Union[int, slice]):
        super().__init__()
        self._idx = idx

    def forward(self, x):
        return x[self._idx]


class Loop(Node):

    def adv(self) -> bool:
        if not self.is_end():
            self._x.adv()
            return True

        return False

    def to(self, nn_module, info):
        if self.y is UNDEFINED:
            return Layer(
                nn_module, x=self, info=info
            )
        return Layer(
            nn_module, x=self.y, info=info
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