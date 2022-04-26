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


class Feedback(object):

    # use to get function type in feedback
    def _f():
        pass

    def __init__(self, key, delay: int=0):

        self._key = key
        self._delay = delay
        self._responses = [None] * (delay + 2)
        self._delay_i = delay
        self._cur_i = 0
        self._set = False
    
    def adv(self):
        self._responses = self._responses[1:] + [None]
        self._delay_i = max(self._delay_i - 1, 0)
        self._cur_i += 1
        self._set = False

    def set(self, node, delay):
        if delay != self._delay:
            raise ValueError(f"Delay value {delay} does not match with delay for feedback {self._delay}")
        if self._set:
            raise ValueError(f"Feedback already set for node {self._key}")
        self._responses[self._delay + 1] = node
        self._set = True

    def get(self, default):
        
        if self._cur_i < self._delay:
            return default() if isinstance(default, type(self._f)) else default
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

    def __getitem__(self, idx: int):
        return self.get(idx)

    # def filter(self, layer):
    #     return Filter(
    #         layer, x=self.y
    #     )

    # # TODO: Reconsider if this is really how I want to do it
    # # probably best just to have another "in" node
    # def empty(
    #     self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
    #     info: Info=None
    # ):
    #     # 
    #     return Layer(
    #         NoArg(nn_module), x=self.y, info=info
    #     )


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

            # Think whether to keep this
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
    def is_parent(self):
        return isinstance(self.op, Tako)

    def children(self) -> typing.Iterator:
        if self.is_parent:
            for layer in self.op.forward_iter(In(self._x)):
                yield layer


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



class LambdaProcess(Process):

    def __init__(self, f):
        self._f = f

    def apply(self, node: Node):
        self._f(node)


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



# DECIDE Later if i need filter


# class NoArg(nn.Module):

#     def __init__(self, module: nn.Module):

#         super().__init__()
#         self.module = module

#     def forward(self, *x):
#         return self.module()


# class Feedback(object):

#     def __init__(self):
#         self._storage: typing.Dict[str, NodeFeedback] = {}
    
#     def adv(self):
#         for node_feedback in self._storage.values():
#             node_feedback.adv()

#     def set(self, key, node: Node, delay=0):

#         if key not in self._storage:
#             self._storage[key] = NodeFeedback(key, delay)

#         self._storage[key].set(node, delay)

#     def get(self, key, default=None):

#         if key in self._storage:
#             return self._storage[key].get()
        
#         return default() if isinstance(default, type(_f)) else default



# class Out(object):

#     def __init__(self):
#         pass

#     def add(self, node):
#         pass

#     def adv(self):
#         pass


# class Filter(Node):
#     # for setting up IIR/FIR Filter
#     # iterator is a "module" that loops over

#     def __init__(self, f: typing.Callable[[Node, NodeSet, Feedback], typing.Iterator], x=UNDEFINED, info: Info=None):
        
#         super().__init__(x, info)
#         self._f = f

#     @property
#     def y(self):

#         # 1) set up in_
#         # 2) set up out
#         # 3) 
#         outs = []
#         feedback = Feedback()
#         for x_i in self._x:
#             # create in_i
#             out = set()
#             self._f(x_i, out, feedback)
#             outs.append(out)
#         # self._y will be based on out
#         self._y = []

#     def children(self) -> typing.Iterator:
#         return super().children()


# want two types of "in" nodes
# one that has a module and no x and
# one that has an x and no module
# figure out how to organize these

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



# class Accumulator(Node):
#     '''
#     Use with loop to accumulate the nodes

#     Need to figure out how to handle 'incoming' properly with this..
#     Probably needs a different kind of 'incoming' in order to perform the loop
#     '''

#     def __init__(self, x=UNDEFINED, incoming: Incoming=None, is_outgoing: bool=False, info: Info=None):
#         super().__init__(is_outgoing, incoming, info)
#         self._x = [x]
    
#     def join(self, node: Node):

#         self._incoming.add(node)
#         self._x.append(node.y)
#         self._y = UNDEFINED

#     @property
#     def y(self):

#         if self._y == UNDEFINED and self._x != UNDEFINED:
#             self._y = [node.y for node in self._nodes]

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


