from abc import ABC, abstractmethod, abstractproperty
from .nodes import UNDEFINED, Node, Layer, In, NodeSet
import typing
import torch.nn as nn
from .utils import ID


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
