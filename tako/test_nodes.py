from uuid import UUID, uuid1

import pytest
from .modules import Null
from . import nodes
import torch.nn as nn
import torch as th
import torch.functional as F
import torch.nn.functional as FNN

# TODO: Still need to test Info, ID etc

class NoArg(nn.Module):

    x = th.rand(2)

    def forward(self):
        return self.x


class TestID:

    def test_id_is_uuid(self):
        id = nodes.ID()
        assert isinstance(id.x, UUID)


class TestLayer:

    def test_x_is_undefined_if_not_set(self):

        layer = nodes.Layer(nn.Sigmoid())
        assert layer.x == nodes.UNDEFINED

    def test_y_is_undefined_if_not_set(self):

        layer = nodes.Layer(nn.Sigmoid())
        assert layer.y == nodes.UNDEFINED

    def test_y_is_defined_if_set(self):
        x = th.rand(2, 2)
        layer = nodes.Layer(nn.Sigmoid())
        layer.x = x
        assert (layer.y == th.sigmoid(x)).all()

    def test_y_is_defined_if_list(self):
        x = th.rand(2, 2)
        layer = nodes.Layer([nn.Sigmoid(), nn.Tanh()])
        layer.x = x
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_y_is_defined_if_empty(self):
        layer = nodes.Layer(NoArg())
        layer.x = nodes.EMPTY
        assert layer.y.size() == th.Size([2])

    def test_mod_is_not_parent(self):
        layer = nodes.Layer(nn.Sigmoid())
        assert not layer.is_parent

    def test_join_combines_two_nodes(self):
        x1 = th.rand(2)
        x2 = th.rand(3)
        layer = nodes.Layer(Null(), x=x1)
        layer2 = nodes.Layer(Null(), x=x2)
        layer3 = layer.join(layer2)

        assert (layer3.y[0] == x1).all()

    def test_join_outputs_undefined_if_one_undefined(self):
        x1 = th.rand(2)
        layer = nodes.Layer(Null(), x=x1)
        layer2 = nodes.Layer(Null())
        layer3 = layer.join(layer2)

        assert layer3.y == nodes.UNDEFINED

    def test_is_parent_with_sequence(self):

        sequence = nodes.Sequence([nn.Sigmoid(), nn.Tanh()])
        layer = nodes.Layer(sequence)
        assert layer.is_parent

    def test_sub_iterates_sequence(self):

        sequence = nodes.Sequence([nn.Sigmoid(), nn.Tanh()])
        layer = nodes.Layer(sequence)
        iter_ =  layer.sub()
        layer = next(iter_)
        layer2 = next(iter_)
        assert isinstance(layer.op, nn.Sigmoid)
        assert isinstance(layer2.op, nn.Tanh)

    def test_layer_sub_does_not_iterate_non_tako(self):

        layer = nodes.Layer(Null())
        with pytest.raises(StopIteration):
            next(layer.sub())

    def test_empty_method_outputs_empty(self):

        layer = nodes.Layer(Null()).empty(NoArg())
        assert (layer.y == NoArg.x).all()


class TestIn:

    def test_x_is_undefined_if_not_set(self):

        in_ = nodes.In()
        assert in_.x == nodes.UNDEFINED

    def test_y_is_correct_value(self):

        x = th.rand(2)
        in_ = nodes.In(x=x)
        assert in_.y is x

    def test_to_is_passed_to_layer(self):

        x = th.rand(2)
        layer = nodes.In(x=x).to(nn.Sigmoid())
        assert (layer.y == th.sigmoid(x)).all()

    def test_to_is_undefined_using_to_layer(self):

        layer = nodes.In().to(nn.Sigmoid())
        assert layer.y == nodes.UNDEFINED

    def test_to_works_with_multiple_modules(self):

        x = th.rand(2)
        layer = nodes.In(x).to(nn.Sigmoid()).to(nn.Tanh())
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_index_works_with_list(self):

        x = [th.rand(2), th.rand(3)]
        layer = nodes.In(x)[0]
        assert (x[0] == layer.y).all()


class TestSequence:

    def test_forward_iter_returns_all_values(self):

        seq = nodes.Sequence([nn.Sigmoid(), nn.Tanh()])
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
