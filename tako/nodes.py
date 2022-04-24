from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from mailbox import NotEmptyError
from re import X
import typing
from numpy import choose, isin
import torch as th
import torch.nn as nn
from functools import partial, singledispatch, singledispatchmethod
import uuid
import time
from .modules import F


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


def if_true(val, obj):
    if val is True:
        return obj


class Incoming(object):

    def __init__(self, incoming=[]):

        self._is_list = isinstance(incoming, list)
        self._incoming = incoming
        self._defined = incoming is None

    def add(self, node):

        self._incoming.append(node)


class NullIncoming(Incoming):

    def __init__(self, incoming):
        super().__init__()

    def add(self, node):
        pass


class Node(ABC):

    def __init__(
        self, x=UNDEFINED, info: Info=None
    ):
        self._info = info or Info()
        self._x = x
        self._y = UNDEFINED
        # self.is_outgoing = is_outgoing
        # if incoming is None:
        #     incoming = NullIncoming()
        # elif not isinstance(incoming, Incoming):
        #     incoming = Incoming(incoming)
        
        # self._incoming = incoming

    @abstractproperty
    def y(self):
        raise NotImplementedError

    @y.setter
    def y(self, y):
        raise NotImplementedError

    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x
    
    def filter(self, layer):
        incoming = if_true(self.is_outgoing, self)
        return Filter(
            layer, x=self.y, incoming=incoming, is_outgoing=self.is_outgoing
        )

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

    def check_id(self, id: ID):
        if self._info.id is None:
            return False
        return self._info.id == id

    def sub(self) -> typing.Iterator:
        if self.is_parent:
            for layer in self._module.forward_iter(In(self._x)):
                yield layer

    def join(self, *others, info: Info=None, join_f=require):
        is_outgoing = self.is_outgoing
        ys = [self.y]
        for other in others:
            is_outgoing = is_outgoing or other.is_outgoing
            ys.append(other.y)
        
        # TODO: Finish.. Need to set up incoming correctly.. Multiple incoming
        if is_outgoing:
            incoming = [self, *others]
        else:
            incoming = None
        
        return Layer(
            nn_module=F(join_f),
            x=ys, 
            incoming=incoming, is_outgoing=is_outgoing, info=info
        )

    def route(self, cond_node):

        if self.is_outgoing or cond_node.is_outgoing:
            is_outgoing = True
            incoming = [cond_node, self]
        else:
            is_outgoing = False
            incoming = None

        return Layer(
            Success(), x=[cond_node.y, self.y], incoming=incoming, is_outgoing=is_outgoing
        ), Layer(
            Failure(), x=[cond_node.y, self.y], incoming=incoming, is_outgoing=is_outgoing
        )

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
        x=UNDEFINED, info: Info=None, 
    ):
        super().__init__(x=x, info=info)
        if isinstance(nn_module, typing.List):
            nn_module = nn.Sequential(*nn_module)
        self.op = nn_module

    @property
    def y(self):

        if self._y == UNDEFINED and self._x != UNDEFINED:

            if isinstance(self._x, Iterator):
                self._y = self._eval_iterator(self._x)
            else:
                self._y = self._eval(self._x)

        return self._y
    
    def _eval_iterator(self, x):
        x: Iterator = x
        y = []
        while True:
            if x.cur == UNDEFINED:
                break
            y.append(self._eval(x.cur))
            x.adv()
            x.respond(y[-1])
        return y

    def _eval(self, x):
        return self.op(x) if x != EMPTY else self.op()

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
        return isinstance(self.op, Tako)

    def sub(self) -> typing.Iterator:
        if self.is_parent:
            for layer in self.op.forward_iter(In(self._x)):
                yield layer


# want two types of "in" nodes
# one that has a module and no x and
# one that has an x and no module
# figure out how to organize these

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


class Filter(Node):
    # for setting up IIR/FIR Filter
    # iterator is a "module" that loops over

    def __init__(self, iterator, x=UNDEFINED):
        pass

    def adv(self):
        pass

    def feedback(self, node):
        # allows for feedback loop
        pass

    def is_end(self):
        pass

    def x_i(self):
        # output the ith element of the filter
        pass


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


    def probe_ys(self, ys: typing.List[ID], by: typing.Dict[typing.ID, typing.Any]):

        result = {y: UNDEFINED for y in ys}

        for layer in self.forward_iter():
            for id, x in by.items():
                if layer.check_id(id):
                    layer.x = x
            
            for y in ys:
                if layer.check_id(y):
                    result[y] = layer.y
        return list(result.value())

    def probe(self, y: ID, by: typing.Dict[typing.ID, typing.Any]):

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
    for layer_i in layer.sub():
        if layer_i.is_parent:
            for layer_j in layer_dive(layer_i):
                yield layer_j
        else: yield layer_i


def dive(tako: Tako, in_):
    for layer in tako.forward_iter(in_):
        yield layer_dive(layer)


# class Loop(Node):

#     def __init__(self, layer: Layer, x=UNDEFINED, incoming=None, is_outgoing: bool=False, info: Info=None):
#         super().__init__(is_outgoing, incoming, info)
#         self._layer = layer
#         self._x = x
#         self._incoming = incoming
#         self._is_outgoing = is_outgoing
    
#     @property
#     def y(self):

#         if self._y == UNDEFINED and self._x != UNDEFINED:

#             self._layer.x = self.x
#             self._y = self._layer.y

#         return self._y

#     @y.setter
#     def y(self, y):
#         self._y = y

#     @property
#     def x(self):
#         return self._x

#     @x.setter
#     def x(self, x):
#         self._x = x

#     @property
#     def is_parent(self):
#         return isinstance(self.op, Tako)

#     def sub(self) -> typing.Iterator:
#         if False:
#             yield True


class Accumulator(Node):
    '''
    Use with loop to accumulate the nodes

    Need to figure out how to handle 'incoming' properly with this..
    Probably needs a different kind of 'incoming' in order to perform the loop
    '''

    def __init__(self, x=UNDEFINED, incoming: Incoming=None, is_outgoing: bool=False, info: Info=None):
        super().__init__(is_outgoing, incoming, info)
        self._x = [x]
    
    def join(self, node: Node):

        self._incoming.add(node)
        self._x.append(node.y)
        self._y = UNDEFINED

    @property
    def y(self):

        if self._y == UNDEFINED and self._x != UNDEFINED:
            self._y = [node.y for node in self._nodes]

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
        return isinstance(self.op, Tako)

    def sub(self) -> typing.Iterator:
        if False:
            yield True



class Iterator(object):

    @abstractmethod
    def iterator(self):
        raise NotImplementedError
    
    @abstractmethod
    def adv(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def is_end(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    @abstractmethod
    def respond(self, y):
        raise NotImplementedError


class ListIterator(Iterator):

    def __init__(self, l: list):
        self._list = l
        self._idx = 0
    
    def adv(self) -> bool:
        if self._idx < len(self._list):
            self._idx += 1
            return True
        return False
    
    def is_end(self) -> bool:
        return self._idx == len(self._list) - 1
    
    def cur(self):
        if self.is_end(): return UNDEFINED
        return self._list[self._idx]
    
    def response(self, y):
        pass




# route
# route = layer.route()
# with route.if_(<condition>) as if_:
#    if_.to()
# with route.else_() as else_:

# class If_(Node):

#     def __init__(self):
#         pass


#     def forward(self):
#         if x[0] is False:
#             return UNDEFINED
#         return x[1]


# def else_if_(x):
#     if x[0] != UNDEFINED:
#         return UNDEFINED
#     elif x[1] is False:
#         return UNDEFINED

#     return x[2]


# def else_(x):

#     if x[0] != UNDEFINED:
#         return UNDEFINED
#     return x[1]



# if_ = x.if_(lambda x)


# else_if = if_.else_if

# with router.if_(lambda x: ) as if_:

#     pass

# with router.else_if(lambda x: ) as else_if:
#     pass




# class Route(Node):
    
#     def __init__(self, cases: typing.List[Case], default: Node=None, x=UNDEFINED, incoming: Node=None, is_outgoing: bool=False, info: Info=None):
#         super().__init__(is_outgoing, incoming, info)
#         self.cases = cases
#         self.default = default
#         self._info = info
#         self._y = UNDEFINED

#     @property
#     def y(self):
        
#         if self._y != UNDEFINED or self.x == UNDEFINED:
#             return self._y
        
#         y = None, None
#         for i, case in enumerate(self.cases):
#             case.x = self.x
#             if case.y[0] is True:
#                 y = i, case.y[1]
#                 break
#         if self.default:
#             self.default.x = self.x
#             y = -1, self.default.y
#         self._y = y
#         return y

#     @y.setter
#     def y(self, y):
#         self._y = y

#     @property
#     def x(self):
#         return self._x

#     @x.setter
#     def x(self, x):
#         self._x = x

#     def is_parent(self):
#         raise NotImplementedError

#     def sub(self) -> typing.Iterator:
#         pass




# class Loop(Node):

#     def __init__(self, iterator, process: Node, x=UNDEFINED, incoming: Node=None, is_outgoing: bool=False, info: Info=None):
#         super().__init__(is_outgoing, incoming, info)
#         self._iterator = iterator
#         self._process = process
#         self._x = x
#         self._y = UNDEFINED

#     @property
#     def y(self):
        
#         if self._y != UNDEFINED or self.x == UNDEFINED:
#             return self._y

#         self._iterator.reset()
#         ys = []
#         self._iterator.x = self._x
#         while True: 
#             self._process.x = self._iterator.cur.x
#             ys.append(self._process.y)
#             if self._iterator.adv() == self._iterator.end(): break
#         self._y = ys
#         return ys

#     @y.setter
#     def y(self, y):
#         self._y = y

#     @property
#     def x(self):
#         return self._x

#     @x.setter
#     def x(self, x):
#         self._x = x

#     def is_parent(self):
#         raise NotImplementedError

#     def sub(self) -> typing.Iterator:
#         pass
#         # if self.is_parent:
#         #     for layer in self._module.forward_iter(In(self._x)):
#         #         yield layer
