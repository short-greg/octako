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


class _UNDEFINED:

    def __str__(self):
        return "UNDEFINED"


UNDEFINED = _UNDEFINED()


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


@dataclass
class Info:
    tags: typing.List[str] = None
    annotation: str = None


class Feedback(object):
    """
    Data structure for storing feedback into the network
    Use to implement recurrency
    """
    def _f():
        pass

    def __init__(self, key: str, delay: int=0):
        """initializer

        Args:
            key (str): 
            delay (int, optional): The amount to delay the feedback by. Defaults to 0.
        """

        self._key = key
        self._delay = delay
        self._responses = [None] * (delay + 2)
        self._set = [False] * (delay + 2)
        self._delay_i = delay
        self._cur_i = 0
    
    def adv(self):
        """Advance the feedback to the next position
        """
        self._responses = self._responses[1:] + [None]
        self._set = self._set[1:] + [False]
        self._delay_i = max(self._delay_i - 1, 0)
        self._cur_i += 1

    def set(self, value, adv=False):
        """Set the currnt value

        Args:
            value : Value to set the feedback to 
            adv (bool, optional): Whether to advance the feedback. Defaults to False.
        """
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
    """Whehter an input is defined
    """
    return not isinstance(x, Node) and x != UNDEFINED


def get_x(node):
    """Get the value of x
    """
    if isinstance(node.x, Node): return UNDEFINED
    return node.x


def to_incoming(node):
    """Return y if node is undefined otherwise return the node
    """

    if node.y is UNDEFINED:
        return node
    return node.y


class Node(ABC):
    """Base class for all network nodes
    """

    def __init__(
        self, x=UNDEFINED, name: str=None, info: Info=None
    ):
        """initializer

        Args:
            x (optional): The input to the node. Defaults to UNDEFINED.
            name (str, optional): The name of the node. Defaults to None.
            info (Info, optional): Infor for the node. Defaults to None.
        """
        self._name = name or str(id(self))
        self._info = info or Info()
        self._outgoing = []
        self._x = UNDEFINED
        self.x = x
    
    @property
    def name(self) -> str:
        """

        Returns:
            str: Name of the ndoe
        """
        
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @abstractproperty
    def y(self):
        """
        Returns:
            Node output if defined else Undefined
        """
        raise NotImplementedError

    @y.setter
    def y(self, y):
        """
        Args:
            y (): The output for the node
        """
        raise NotImplementedError

    @property
    def x(self):
        """

        Returns:
             The input into the network
        """
        if isinstance(self._x, Node):
            return UNDEFINED
        return self._x

    @x.setter
    def x(self, x):
        """
        Args:
            x (): Set the input to the network
        """
        if isinstance(self._x, Node):
            self._x._remove_outgoing(self)
        if isinstance(x, Node):
            x._add_outgoing(self)
        self._x = x

    def lookup(self, by: dict):
        """
        Args:
            by (): _description_

        Returns:
            _type_: _description_
        """
        return by.get(self._name, UNDEFINED)
    
    def set_by(self, by, y):
        """ Set the value of y

        Args:
            by (dict): 
            y (): The output to the network
        """
        by[self.id] = y
    
    def _add_outgoing(self, node):
        node._outgoing.append(self)

    def _remove_outgoing(self, node):
        node._outgoing.remove(self)

    def to(
        self, nn_module: typing.Union[typing.List[nn.Module], nn.Module], 
        name: str=None, info: Info=None
    ):
        """Connect the layer to another layer

        Args:
            nn_module (typing.Union[typing.List[nn.Module], nn.Module]): Module
            name (str, optional): Name of the node. Defaults to None.
            info (Info, optional): Info for the layer. Defaults to None.

        Returns:
            Layer: 
        """
        return Layer(
            nn_module, x=to_incoming(self), name=name, info=info
        )

    def join(self, *others, info: Info=None):
        """_summary_

        Args:
            *others: nodes to join with
            info (Info, optional): _description_. Defaults to None.

        Returns:
            Joint
        """
        
        ys = []
        for node in [self, *others]:
            ys.append(to_incoming(node))

        return Joint(
            x=ys, info=info
        )

    def route(self, cond_node):
        """

        Args:
            cond_node (_type_): _description_

        Returns:
            _type_: _description_
        """

        return Decision(
            to_incoming(cond_node), to_incoming(self)
        ), Decision(
            to_incoming(cond_node), to_incoming(self), positive=False
        )

    def get(self, idx: typing.Union[slice, int], info: Info=None):
        """Get an index from the output

        Args:
            idx (typing.Union[slice, int]): 
            info (Info, optional): _description_. Defaults to None.

        Returns:
            Index
        """
        return Index(
            idx, x=to_incoming(self), info=info
        )
    
    def loop(self, f, *aggregators, info: Info=None):
        """Loop over 

        Args:
            f (): Iterator function
            info (Info, optional): Info. Defaults to None.

        Returns:
            Loop
        """

        iterator = Iterator(x=to_incoming(self), info=info)
        
        accumulators = []
        for aggregator in aggregators:
            accumulators.append(iterator.aggregate(aggregator))
        while True:
            layers = f(iterator)
            if isinstance(layers, Node):
                accumulators[0].from_(layers)
            else:
                for layer, accumulator in zip(layers, accumulators):
                    accumulator.from_(layer)
            iterator.adv()
            if iterator.is_end():
                break
        return accumulators

    def __getitem__(self, idx: int):
        return self.get(idx)
    
    @abstractmethod
    def _probe_out(self, by: dict):
        raise NotImplementedError

    def probe(self, by):
        """Probe the layer

        Args:
            by (dict)

        Returns:
            result: valeu 
        """
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
        """initializer

        Args:
            cond_x (optional): _description_. Defaults to UNDEFINED.
            x (optional): Input into the decision node. Defaults to UNDEFINED.
            positive (bool, optional): Whether the node excutes on positive. Defaults to True.
            name (str, optional): Name of the decision node. Defaults to None.
            info (Info, optional): Info for the ndoe. Defaults to None.
        """
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
        """initializer

        Args:
            x (_type_, optional): Input to the node. Defaults to UNDEFINED.
            name (str, optional): Name of the node. Defaults to None.
            info (Info, optional): Info for the node. Defaults to None.
        """
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
        """
        Returns:
            The input to the node
        """
        
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
            xs.append(x_i)
        self._x = xs

    def _probe_out(self, by):
        y = []
        for x_i in self._x:
            if is_defined(x_i):
                y.append(x_i)
            else:
                y.append(x_i.probe(by))
        return y
                

class Index(Node):
    """

    Args:
        Node (_type_): _description_
    """
    
    def __init__(self, idx: typing.Union[int, slice], x=UNDEFINED, name: str=None, info: Info=None):
        super().__init__(x, name, info)
        self._idx = idx

    @property
    def y(self):
        """

        Returns:
            The output of the node
        """
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
        """initializer

        Args:
            nn_module (typing.Union[typing.List[nn.Module], nn.Module]): Module for the layer
            x (optional): Input to the node. Defaults to UNDEFINED.
            name (str, optional): Name of the node. Defaults to None.
            info (Info, optional): Info for the node. Defaults to None.
        """
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
            print(self._x)
            self._y = self.op(self._x)

        return self._y
    
    @y.setter
    def y(self, y):
        """
        Args:
            y (): The output of the node
        """
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

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._x

    @y.setter
    def y(self, y):
        self._x = y

    def _probe_out(self, by):
        return self._x


class Iterator(Node):

    def __init__(
        self, x=UNDEFINED, name: str=None, info: Info=None
    ):
        """initializer

        Args:
            x (_type_, optional): The input. Defaults to UNDEFINED.
            name (str, optional): Name of the iterator. Defaults to None.
            info (Info, optional): Info for the iterator Defaults to None.
        """
        self._cur = None
        self._end = False
        super().__init__(x, name, info)

    def adv(self) -> bool:
        """advance the iterator forward

        Returns:
            bool: Whether in progress (True) or at the end (False)
        """

        if self._x is UNDEFINED:
            self._end = True

        if self.is_end():
            return False
        try:
            self._cur = next(self._iter)
        except StopIteration:
            self._cur = UNDEFINED
            self._end = True

        return True

    def to(self, nn_module, name: str=None, info: Info=None) -> Layer:
        """

        Args:
            nn_module (_type_): 
            name (str, optional):  Defaults to None.
            info (Info, optional):  Defaults to None.

        Returns:
            Layer
        """
        return Layer(
            nn_module, x=to_incoming(self), name=name, info=info
        )
    
    def aggregate(self, aggregator, info: Info=None):
        """_summary_

        Args:
            aggregator (): 
            info (Info, optional): Info . Defaults to None.

        Returns:
            Accumulator
        """
        return Accumulator(
            self, aggregator, info=info
        )
    
    @property
    def x(self):
        return self._x if is_defined(self._x) else UNDEFINED

    @x.setter
    def x(self, x):
        self._cur = None
        self._end = False
        self._x = x
        self._iter = iter(x)
        self.adv()

    @property
    def y(self):
        if not is_defined(self._cur):
            return UNDEFINED

        return self._cur
        
    def is_end(self):

        return self._end 
    
    def reset(self):
        if is_defined(self._x):
            self._iter = iter(self._x)
            self.adv()
    
    def _probe_out(self, by):
        if is_defined(self._x):
            return self._cur
        cur = by.get(self._name)
        if cur is not None:
            return cur

        x = self._x.probe(by)
        if x is UNDEFINED:
            return UNDEFINED
        
        try:
            cur = next(iter(x))
            by[self.name] = cur
            return cur
        except StopIteration:
            by[self.name] = UNDEFINED
            return UNDEFINED


class Aggregator(nn.Module):

    @abstractmethod
    def update(self, x, state=None):
        pass

    @abstractmethod
    def forward(self, state=None):
        pass

    @abstractmethod
    def spawn(self):
        pass


class All(Aggregator):
    """Retrieve all outputs
    """

    def update(self, x, state=None):
    
        if state is not None:
            return [*state, x]
        return [x]
    
    def forward(self, state):
        return th.stack(state)

    def spawn(self):
        return All()


class Last(Aggregator):
    """Retrieve the last output
    """

    def update(self, x, state=None): 
        return x
    
    def forward(self, state):
        return state

    def spawn(self):
        return Last()


class Accumulator(Node):

    def __init__(self, loop: Iterator, aggregator: Aggregator, x=UNDEFINED, name: str=None, info=None):
        """
        Args:
            loop (Iterator): 
            aggregator (Aggregator): 
            x (optional): Input to the accumulator. Defaults to UNDEFINED.
            name (str, optional): Name of the node. Defaults to None.
            info (_type_, optional): Info for the node. Defaults to None.
        """
        super().__init__(x, name, info)
        self._loop = loop
        self._aggregator: Aggregator = aggregator
        self._state = None
    
    def from_(self, node: Node):
        if is_defined(node.y):
            self._state = self._aggregator.update(node.y, self._state)
        else:
            self.x = node

    @property
    def y(self):
        
        if self.x is not UNDEFINED:
            self._state = self._aggregator(self.x, self._state)
        if self._state is not None:
            self._y = self._aggregator(self._state)
        else:
            self._y = UNDEFINED
        return self._y

    def _probe_out(self, by):
        if is_defined(self._x):
            return self._aggregator(self._x)
        
        if self._x == UNDEFINED:
            return UNDEFINED
        
        aggregator = self._aggregator.spawn()
        state = None

        while True:
            x = self._x.probe(by)

            if x is UNDEFINED:
                break

            state = aggregator.update(x, state)

            cur_it = by[self._loop.name]
            if cur_it is UNDEFINED:
                break
            
            # update the current iterator position
            # and store so it will be advanced for the
            # next probe
            try:
                cur_it = next(cur_it)
                by[self._loop.name] = cur_it
            except StopIteration:
                break

        if state is None:
            return UNDEFINED

        return aggregator(state)


class Process(ABC):

    @abstractmethod
    def apply(self, node: Node):
        pass


class NodeSet(object):
    """Set of nodes. Can use to probe
    """

    def __init__(self, nodes: typing.List[Node]):
        self._nodes = {node.name: node for node in nodes}
    
    def apply(self, process: Process):
        """Apply a process on each node in the set

        Args:
            process (Process)
        """
        for node in self._nodes.values():
            if isinstance(process, Process):
                process.apply(node)
            else:
                process(node)
        
    def __getitem__(self, key: str) -> Node:
        """
        Args:
            key (str): name of the nodeset

        Returns:
            Node
        """
        if key not in self._nodes:
            raise KeyError("There is no node named key.")
        return self._nodes[key]

    def probe(self, by):
        """
        Args:
            by (dict): Outputs for nodes {'node': {output}}

        Returns:
            typing.Union[typing.List[torch.Tensor], torch.Tensor]
        """
        result = []
        for node in self._nodes:
            result.append(node.probe(by))
        return result


class LambdaProcess(Process):

    def __init__(self, f):
        self._f = f

    def apply(self, node: Node):
        self._f(node)


class Tako(nn.Module):

    @abstractmethod
    def forward_iter(self, in_: Node=None) -> typing.Iterator:
        pass

    def sub(self, y: typing.Union[str, typing.List[str]], x: typing.Union[str, typing.List[str]]):
        """
        Extract a sub network

        TODO: Simplify the code 
        """

        x_is_list = isinstance(x, list)
        y_is_list = isinstance(y, list)

        if y_is_list:
            out = {y_i: UNDEFINED for y_i in y}
        else:
            out = UNDEFINED
        
        if x_is_list:
            found = [False] * len(x)
        else:
            found = False

        in_ = In()
        
        for layer in self.forward_iter(in_):
            if x_is_list and layer.name in x:
                idx = x.index(layer.name)
                in_[idx].rewire(layer)
                found[idx] = True
            elif not x_is_list and layer.name == x:
                in_.rewire(layer)
                found = True
            
            if y_is_list and layer.name in y:
                out[layer.name] = layer
        
        if (x_is_list and False in found) or (not x_is_list and found is False):
            raise RuntimeError()

        if (y_is_list and UNDEFINED in out.values()) or (not y_is_list and out is UNDEFINED):
            raise RuntimeError()
        elif y_is_list:
            return list(out.values())
        return out      

    def probe(self, y: typing.Union[str, typing.List[str]], in_: Node=None, by: typing.Dict[str, typing.Any]=None):
        
        by = by or {}
        if isinstance(y, list):
            out = {y_i: UNDEFINED for y_i in y}
            is_list = True
        else:
            is_list = False
            out = UNDEFINED

        for layer in self.forward_iter(in_):
            for id, x in by.items():
                if layer.name == id:
                    layer.y = x
            
            if is_list and layer.name in out:
                out[layer.name] = layer.y
            elif not is_list:
                if layer.name == y:
                    return layer.y

        if is_list:
            return list(out.values())
        return out

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


class Filter(ABC):

    @abstractmethod
    def check(self, layer: Layer) -> bool:
        pass

    def extract(self, tako: Tako) -> NodeSet:
        return NodeSet(
            [layer for layer in self.filter(tako) if self.check(layer)]
        )
    
    def apply(self, tako: Tako, process: Process):
        for layer in self.filter(tako):
            if self.check(layer):
                process.apply(layer)

    def filter(self, tako) -> typing.Iterator:
        for layer in tako.forward_iter():
            if self.check(layer):
                yield layer


class TagFilter(Filter):

    def __init__(self, filter_tags: typing.List[str]):

        self._filter_tags = set(filter_tags)

    def check(self, layer: Layer) -> bool:
        return len(self._filter_tags.intersect(layer.info.tags)) > 0


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
        in_: typing.Union[ID, typing.List[str]], 
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


class BinaryClassify(nn.Module):

    def __init__(self, threshold: float=0.5):

        super().__init__()
        self.threshold = threshold

    def forward(self, x: th.Tensor):
        return (x > self.threshold).float()


class Classify(nn.Module):

    def __init__(self, dim: int=-1):

        super().__init__()
        self.dim = dim

    def forward(self, x: th.Tensor):
        return th.argmax(x, dim=self.dim)
