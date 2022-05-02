
from . import nodes
from . import tako 
import torch as th
import torch.nn as nn

class NoArg(nn.Module):

    x = th.rand(2)

    def forward(self):
        return self.x


class TestSequence:

    def test_forward_iter_returns_all_values(self):

        seq = tako.Sequence([nn.Sigmoid(), nn.Tanh()])
        x = th.rand(2)
        in_ = nodes.In(x)
        iter_ = seq.forward_iter(in_)
        layer = next(iter_)
        assert (layer.y == th.sigmoid(x)).all()
        layer = next(iter_)
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_forward_iter_returns_undefined(self):

        seq = nodes.Sequence([NoArg()])
        in_ = nodes.In(nodes.EMPTY)
        iter_ = seq.forward_iter(in_)
        layer= next(iter_)
        assert (layer.y == NoArg.x).all()

