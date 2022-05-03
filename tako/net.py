from abc import ABC, abstractmethod, abstractproperty
import typing
import torch as th
import torch.nn as nn

from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from re import X
import typing
from numpy import choose, isin
from functools import partial, singledispatch, singledispatchmethod
import uuid
import time
from typing import Any
import typing
from functools import partial

import uuid


class ID(object):

    def __init__(self, id: uuid.UUID=None):

        self.x = id if id is not None else uuid.uuid4()


UNDEFINED = object()

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




@dataclass
class Info:
    tags: typing.List[str] = None
    annotation: str = None


class Feedback(object):
    """

    """
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
    return not isinstance(x, Node) and x != UNDEFINED


def get_x(node):
    if isinstance(node.x, Node): return UNDEFINED
    return node.x


def to_incoming(node):

    if node.y is UNDEFINED:
        return node
    return node.y


class Node(ABC):

    def __init__(
        self, x=UNDEFINED, name: str=None, info: Info=None
    ):
        self._name = name or str(id(self))
        self._info = info or Info()
        self._outgoing = []
        self._x = UNDEFINED
        self.x = x

    @abstractproperty
    def y(self):
        raise NotImplementedError

    @y.setter
    def y(self, y):
        raise NotImplementedError

    @property
    def x(self):
        if isinstance(self._x, Node):
            return UNDEFINED
        return self._x

    @x.setter
    def x(self, x):
        if isinstance(self._x, Node):
            self._x._remove_outgoing(self)
        if isinstance(x, Node):
            x._add_outgoing(self)
        self._x = x

    def lookup(self, by):
        value = by.get(self._name, UNDEFINED)
    
    def set_by(self, by, y):
        by[self.id] = y
    
    def _add_outgoing(self, node):
        node._outgoing.append(self)

    def _remove_outgoing(self, node):
        node._outgoing.remove(self)

    def to(
        self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
        info: Info=None
    ):
        return Layer(
            nn_module, x=to_incoming(self), info=info
        )

    # TODO: 
    def join(self, *others, info: Info=None, join_f=require):
        
        ys = []
        
        for node in [self, *others]:
            ys.append(to_incoming(node))

        return Joint(
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
        for filter in filters:
            accumulators.append(loop.filter(filter))
        while True:
            layers = f(loop)
            if isinstance(layers, Node):
                accumulators[0].from_(layers)
            else:
                for layer, accumulator in zip(layers, accumulators):
                    accumulator.from_(layer)
            loop.adv()
            if loop.is_end():
                break
        return accumulators

    def __getitem__(self, idx: int):
        return self.get(idx)
    
    @abstractmethod
    def _probe_out(self, by: dict):
        raise NotImplementedError

    def probe(self, by):
        value = self._info.lookup(by)
        if value is not None:
            return value

        result = self._probe_out(by)        

        if len(self._outgoing) > 1:
            self.set_by(by, result)
        return result


class Cur(nn.Module):

    def __init__(self, default_f):
        super().__init__()
        self._default_f = default_f

    def forward(self, x: Feedback):
        return x.get(self._default_f)


def cur(feedback_node: Node, default_f):
    return feedback_node.to(Cur(default_f))


class Decision(Node):

    def __init__(
        self, cond_x=UNDEFINED, x=UNDEFINED, positive: bool=True, name: str=None, info: Info=None
    ):
        super().__init__(x, name=name, info=info)
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

    def _probe_out(self, by):

        if is_defined(self._x) and is_defined(self._cond_x):
            return self.y    
        elif not is_defined(self._x):
            x = self._x.probe(by)
            cond_x = self._cond_x
        elif not is_defined(self._x):
            cond_x = self._cond_x.probe(by)
            x = self._x
        else:
            cond_x = self._cond_x.probe(by)
            x = self._x.probe(by)
        
        if not is_defined(cond_x) or cond_x is not self._positive:
            return UNDEFINED
        elif not is_defined(x):
            return UNDEFINED
        return x


class Joint(Node):

    def __init__(
        self, x=UNDEFINED, name: str=None, info: Info=None
    ):
        super().__init__(x, name=name, info=info)
        self._y = UNDEFINED

    @property
    def y(self):

        undefined = False
        for x_i in self._x:
            if isinstance(x_i, Node):
                undefined = True
        
        if undefined:
            self._y = UNDEFINED
        else:
            self._y = self._x

        return self._y

    @y.setter
    def y(self, y):
        self._y = y
    
    @property
    def x(self):
        
        if self._x is UNDEFINED:
            return UNDEFINED
        x = []
        for x_i in self._x:
            if isinstance(x_i, Node):
                x.append(UNDEFINED)
            else:
                x.append(x_i)

        return x

    @x.setter
    def x(self, x):

        if self._x is not UNDEFINED:
            for x_i in self._x:
                if isinstance(x_i, Node):
                    x_i._remove_outgoing(self)

        xs = []
        for x_i in x:
            if isinstance(x_i, Node):
                x_i._add_outgoing(self)
            xs.append(x)
        self._x = xs

    def _probe_out(self, by):
        
        for x_i in self._x:
            if is_defined(x_i):
                y.append(x_i)
            else:
                y.append(x_i.probe(by))
        return y
                

class Index(Node):
    
    def __init__(self, idx: typing.Union[int, slice], x=UNDEFINED, name: str=None, info: Info=None):
        super().__init__(x, name, info)
        self._idx = idx

    @property
    def y(self):
        if isinstance(self._x, Node):
            return UNDEFINED
        else:
            return self._x[self._idx]
    
    def _probe_out(self, by):

        if is_defined(self._x):
            x = self._x
        else:
            x = self._x.probe(by)
    
        return x[self._idx]


class Layer(Node):

    def __init__(
        self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
        x=UNDEFINED, name: str=None, info: Info=None
    ):
        super().__init__(x, name=name, info=info)
        if isinstance(nn_module, typing.List):
            nn_module = nn.Sequential(*nn_module)
        self.op = nn_module
        self._y = UNDEFINED

    @property
    def y(self):

        if self._y == UNDEFINED and isinstance(self._x, Node):
            return UNDEFINED

        elif self._y == UNDEFINED and self._x != UNDEFINED:
            self._y = self.op(self._x)

        return self._y
    
    @y.setter
    def y(self, y):
        self._y = y
    
    def _probe_out(self, by):
        if is_defined(self._x):
            x = self._x
        else:
            x = self._x.probe(by)
        return self.op(x)


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

    def _probe_out(self, by):
        return self._x


class Loop(Node):

    def adv(self) -> bool:
        if not self.is_end():
            self._x.adv()
            return True

        return False

    def to(self, nn_module, info: Info=None):
        return Layer(
            nn_module, x=to_incoming(self), info=info
        )
    
    def filter(self, filter, info: Info=None):
        return Accumulator(
            self, filter, info=info
        )

    @property
    def y(self):
        if self._x is UNDEFINED or self._x.is_end():
            return UNDEFINED
        return self._x.cur
        
    def is_end(self):
        
        return self._x is UNDEFINED or self._x.is_end()
    
    def reset(self):
        if is_defined(self._x):
            self._x.reset()
    
    def _probe_out(self, by):
        if is_defined(self._x.cur):
            return self._x.cur
        return self._x.cur.probe(by)


class Filter(nn.Module):

    @abstractmethod
    def update(self, x, state=None):
        pass

    @abstractmethod
    def forward(self, x, state=None):
        pass


class All(Filter):

    def update(self, x, state=None):
    
        if state is not None:
            return [*state, x]
        return [x]
    
    def forward(self, state):
        return th.stack(state)


class Last(Filter):

    def update(self, x, state=None): 
        return x
    
    def forward(self, state):
        return state


class Accumulator(Node):

    def __init__(self, loop: Loop, filter: Filter, x=UNDEFINED, name: str=None, info=None):
        super().__init__(x, name, info)
        self._loop = loop
        self._filter: Filter = filter
        self._state = None
        
    def from_(self, node: Node):
        if is_defined(node.y):
            self.x = self._filter.update(node.y, self.x)
        else:
            self.x = node

    @property
    def y(self):
        
        if self._state is not None:
            self._y = self._filter(self._state)
        else:
            self._y = UNDEFINED
        return self._y

    def _probe_out(self, by):
        if is_defined(self._x):
            return self._filter(self._x)
        
        if self._x == UNDEFINED:
            return UNDEFINED
    
        iterator = self.loop.probe(by)
        if iterator is UNDEFINED:
            return UNDEFINED
        
        iterator = iterator.spawn()
        self.loop.set_by(iterator)

        filter = self._filter.spawn()
        state = None
        while not iterator.is_end():
            state = filter.update(self._x.probe(by), state)
            iterator.adv()
        return filter(state)


class Iterator(object):

    def __init__(self, iterable):
        self._iterable = iterable
        self._iter = iter(iterable)
        self._end = False
        self._cur = None
        self.adv()

    def adv(self) -> bool:
        if self._end:
            return False
        try:
            self._cur = next(self._iter)
        except StopIteration:
            self._cur = UNDEFINED
            self._end = True

    def is_end(self) -> bool:
        return self._end

    def reset(self):
        self._iter = iter(self._iterable)
        self.adv()

    @property
    def cur(self):
        return self._cur


class Iterate(nn.Module):

    def forward(self, x) -> Iterator:
        return Iterator(x)


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



class Tako(nn.Module):

    @abstractmethod
    def forward_iter(self, in_: Node) -> typing.Iterator:
        pass

    def probe_ys(self, ys: typing.List[ID], by: typing.Dict[ID, typing.Any]):

        result = {y: UNDEFINED for y in ys}

        for layer in self.forward_iter():
            for id, x in by.items():
                if layer.check_id(id):
                    layer.x = x
            
            for y in ys:
                if layer.check_id(y):
                    result[y] = layer.y
        return list(result.value())

    def probe(self, y: ID, by: typing.Dict[ID, typing.Any]):

        # TODO: do I want to raise an exception if UNDEFINED?
        for layer in self.forward_iter():
            for id, x in by.items():
                if layer.check_id(id):
                    layer.x = x
            
            if layer.check_id(y):
                return layer.y

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


class Find(ABC):

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
    if isinstance(layer.op, Tako):
        for layer_i in layer.op.forward_iter():
            for layer_j in layer_dive(layer_i):
                yield layer_j

    else: yield layer_i


def dive(tako: Tako, in_):
    for layer in tako.forward_iter(in_):
        yield layer_dive(layer)


class Network(nn.Module):

    def __init__(
        self, out: typing.Union[Node, NodeSet], 
        in_: typing.Union[ID, typing.List[ID]], 
        by
    ):
        # specify which nodes to 
        self._out = out
        self._in = in_
        self._by = by
        # counts = outgoingcount(self._out)

    def forward(self, x):
        
        # need to mark which ones
        # i want to store in by
        # by = by.update(**self._in, x)
        # by.outgoing_count(t)

        by = {
            **self._by,
            **zip(self._in, x)
        }
        return self._out.probe(by)


# def first(x):

#     for x_i in x:
#         if x_i != UNDEFINED:
#             return x_i
#     return UNDEFINED


# def require(x):

#     for x_i in x:
#         if x_i == UNDEFINED:
#             return UNDEFINED
#     return x
