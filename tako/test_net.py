import typing
import torch as th
import torch.nn as nn
from functools import partial
from uuid import UUID, uuid1
from . import net
import pytest
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
        id = net.ID()
        assert isinstance(id.x, UUID)


class TestLayer:

    def test_x_is_undefined_if_not_set(self):

        layer = net.Layer(nn.Sigmoid())
        assert layer.x == net.UNDEFINED

    def test_y_is_undefined_if_not_set(self):

        layer = net.Layer(nn.Sigmoid())
        assert layer.y == net.UNDEFINED

    def test_y_is_defined_if_set(self):
        x = th.rand(2, 2)
        layer = net.Layer(nn.Sigmoid())
        layer.x = x
        assert (layer.y == th.sigmoid(x)).all()

    def test_y_is_defined_if_list(self):
        x = th.rand(2, 2)
        layer = net.Layer([nn.Sigmoid(), nn.Tanh()])
        layer.x = x
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_join_combines_two_nodes(self):
        x1 = th.rand(2)
        x2 = th.rand(3)
        layer = net.Layer(net.Null(), x=x1)
        layer2 = net.Layer(net.Null(), x=x2)
        layer3 = layer.join(layer2)

        assert (layer3.y[0] == x1).all()

    def test_y_is_defined_if_gen_and_true_passed(self):
        layer = net.Layer(net.Gen(th.rand, 2, 4))
        layer.x = True
        assert layer.y.size() == th.Size([2, 4])

    def test_y_is_undefined_if_gen_and_false_passed(self):
        layer = net.Layer(net.Gen(th.rand, 2, 4))
        layer.x = False
        assert layer.y is net.UNDEFINED

    def test_join_outputs_undefined_if_one_undefined(self):
        x1 = th.rand(2)
        layer = net.Layer(net.Null(), x=x1)
        layer2 = net.Layer(net.Null())
        layer3 = layer.join(layer2)

        assert layer3.y is net.UNDEFINED
    

class TestIn:

    def test_x_is_undefined_if_not_set(self):

        in_ = net.In()
        assert in_.x == net.UNDEFINED

    def test_y_is_correct_value(self):

        x = th.rand(2)
        in_ = net.In(x=x)
        assert in_.y is x

    def test_to_is_passed_to_layer(self):

        x = th.rand(2)
        layer = net.In(x=x).to(nn.Sigmoid())
        assert (layer.y == th.sigmoid(x)).all()

    def test_to_is_undefined_using_to_layer(self):

        layer = net.In().to(nn.Sigmoid())
        assert layer.y is net.UNDEFINED

    def test_to_works_with_multiple_modules(self):

        x = th.rand(2)
        layer = net.In(x).to(nn.Sigmoid()).to(nn.Tanh())
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_index_works_with_list(self):

        x = [th.rand(2), th.rand(3)]
        layer = net.In(x)[0]
        assert (x[0] == layer.y).all()


class TestRoute:

    def test_route_with_success_outputting_true(self):
        
        cond = net.In(True)
        x = th.rand(2)
        success, failure = net.In(x).route(cond)

        assert (success.y == x).all()
        assert (failure.y == net.UNDEFINED)      

    def test_route_with_failure_outputting_true(self):
        
        cond = net.In(False)
        x = th.rand(2)
        success, failure = net.In(x).route(cond)

        assert (failure.y == x).all()
        assert (success.y == net.UNDEFINED)        

    def test_route_with_undefined_input(self):
        
        cond = net.In(False)
        success, failure = net.In().route(cond)

        assert failure.y is net.UNDEFINED  
        assert failure.x is net.UNDEFINED   


class TestFeedback:

    def test_feedback_get_gets_default(self):

        feedback = net.Feedback('c2', 0)
        result = feedback.get(partial(th.zeros, 2, 2))
        assert (result == th.zeros(2, 2)).all()

    def test_feedback_get_gets_value(self):

        feedback = net.Feedback('c2', 0)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        result = feedback.get(partial(th.zeros, 2, 2))
        assert (result == x).all()

    def test_feedback_get_gets_value_with_delay(self):

        feedback = net.Feedback('c2',1)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        feedback.adv()
        result = feedback.get(partial(th.zeros, 2, 2))
        assert (result == x).all()

    def test_feedback_get_gets_default_if_not_set(self):

        feedback = net.Feedback('c2',0)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        feedback.adv()
        result = feedback.get(partial(th.zeros, 2, 2))
        assert (result == th.zeros(2, 2)).all()

    def test_feedback_in_network(self):

        feedback = net.Feedback('c2', 0)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        in_ = net.In(feedback)
        cur_ = net.cur(in_, partial(th.zeros, 2, 2))
        assert (cur_.y == x).all()

    def test_feedback_in_network_returns_default(self):

        feedback = net.Feedback('c2', 1)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        in_ = net.In(feedback)
        cur_ = net.cur(in_, partial(th.zeros, 2, 2))
        assert (cur_.y == th.zeros(2, 2)).all()

    def test_feedback_retrieval_works_correctly_with_one_delay(self):

        feedback = net.Feedback('c2', 1)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        in_ = net.In(feedback)
        cur_ = net.cur(in_, partial(th.zeros, 2, 2))
        feedback.set(cur_.y, True)
        assert (feedback.get() == x).all()

    def test_updating_feedback_sets_the_feedback(self):

        feedback = net.Feedback('c2', 1)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        in_ = net.In(feedback)
        cur_ = net.cur(in_, partial(th.zeros, 2, 2))
        feedback.set(cur_.y, True)
        feedback.adv()
        assert (feedback.get() == th.zeros(2, 2)).all()


class TestIterator:

    def test_iterator_with_list(self):
        iterator = net.Iterator([2,3])
        assert iterator.y == 2
        iterator.adv()
        assert iterator.y == 3
        iterator.adv()
        assert iterator.is_end() is True

    def test_iterator_end_outputs_undefined(self):
        iterator = net.Iterator([])
        assert iterator.y == net.UNDEFINED
        assert iterator.is_end() is True


def loop_(in_: net.Node):

    return in_.to(nn.Linear(2, 2))


class TestLoop:

    def test_loop_with_successfully_stacks_modules(self):
        
        iterator = net.Iterator([th.rand(2),th.rand(2)])
        in_ = net.In(x=th.randn(3, 2))
        acc, = in_.loop(loop_, net.All())
        assert acc.y.size() == th.Size([3, 2])

    def test_loop_successfully_chooses_last(self):
        iterator = net.Iterator([th.rand(2),th.rand(2)])
        in_ = net.In(x=th.randn(3, 2))
        acc, = in_.loop(loop_, net.Last())
        assert acc.y.size() == th.Size([2])


class NoArg(nn.Module):

    x = th.rand(2)

    def forward(self):
        return self.x


class TestSequence:

    def test_forward_iter_returns_all_values(self):

        seq = net.Sequence([nn.Sigmoid(), nn.Tanh()])
        x = th.rand(2)
        in_ = net.In(x)
        iter_ = seq.forward_iter(in_)
        layer = next(iter_)
        assert (layer.y == th.sigmoid(x)).all()
        layer = next(iter_)
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_forward_iter_returns_undefined(self):

        seq = net.Sequence([net.Gen(NoArg())])
        in_ = net.In(True)
        iter_ = seq.forward_iter(in_)
        layer= next(iter_)
        assert (layer.y == NoArg.x).all()


class TestTako:

    class TakoT(net.Tako):

        X = 'x'
        Y = 'y'

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward_iter(self, in_: net.Node=None) -> typing.Iterator:
            
            in_ = in_ or net.In()
            linear = in_.to(self.linear, name=self.X)
            yield linear
            sigmoid = linear.to(nn.Sigmoid(), name=self.Y)
            yield sigmoid

    def test_probe_linear_outputs_correct_value(self):
        tako = self.TakoT()
        in_ = th.rand(1, 2)
        y = tako.probe(tako.X, in_=net.In(in_))
        assert (y == tako.linear(in_)).all()
    
    def test_probe_sigmoid_outputs_correct_value_with_linear(self):
        tako = self.TakoT()
        in_ = th.rand(1, 2)
        y = tako.probe(tako.Y, by={tako.X: in_})
        assert (y == th.sigmoid(in_)).all()
    
    def test_probe_multiple_outputs_correct_value(self):
        tako = self.TakoT()
        in_ = th.rand(1, 2)
        y1, y2 = tako.probe([tako.Y, tako.X], in_=net.In(in_))
        linear_out = tako.linear(in_)
        sigmoid_out = th.sigmoid(linear_out)
        assert (y1 == sigmoid_out).all()
        assert (y2 == linear_out).all()


class TestNodeSet:

    def test_getindex_with_valid_value(self):
        in_ = net.In(th.randn(2, 2), name='in')
        out = net.Layer(nn.Linear(2, 2), name='Linear')
        node_set = net.NodeSet([in_, out])
        assert node_set['in'] is in_
        assert node_set['Linear'] is out

    def test_getindex_with_valid_value(self):
        in_ = net.In(th.randn(2, 2), name='in')
        out = net.Layer(nn.Linear(2, 2), name='Linear')
        node_set = net.NodeSet([in_, out])
        with pytest.raises(KeyError):
            node_set['x']

    def test_apply_by_appending_to_name(self):
        in_ = net.In(th.randn(2, 2), name='in')
        out = net.Layer(nn.Linear(2, 2), name='Linear')

        def rename(node):
            node.name = node.name + '_1'

        node_set = net.NodeSet([in_, out])
        node_set.apply(rename)
        assert in_.name == 'in_1'
        assert out.name == 'Linear_1'
