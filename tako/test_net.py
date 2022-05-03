
from . import nodes
from . import tako 
import torch as th
import torch.nn as nn
from functools import partial
from uuid import UUID, uuid1

import pytest

from tako.utils import UNDEFINED
from .modules import Null, Gen
from . import tako
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
        id = tako.ID()
        assert isinstance(id.x, UUID)


class TestLayer:

    def test_x_is_undefined_if_not_set(self):

        layer = tako.Layer(nn.Sigmoid())
        assert layer.x == tako.UNDEFINED

    def test_y_is_undefined_if_not_set(self):

        layer = tako.Layer(nn.Sigmoid())
        assert layer.y == tako.UNDEFINED

    def test_y_is_defined_if_set(self):
        x = th.rand(2, 2)
        layer = tako.Layer(nn.Sigmoid())
        layer.x = x
        assert (layer.y == th.sigmoid(x)).all()

    def test_y_is_defined_if_list(self):
        x = th.rand(2, 2)
        layer = tako.Layer([nn.Sigmoid(), nn.Tanh()])
        layer.x = x
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_join_combines_two_nodes(self):
        x1 = th.rand(2)
        x2 = th.rand(3)
        layer = tako.Layer(Null(), x=x1)
        layer2 = tako.Layer(Null(), x=x2)
        layer3 = layer.join(layer2)

        assert (layer3.y[0] == x1).all()

    def test_y_is_defined_if_gen_and_true_passed(self):
        layer = tako.Layer(Gen(th.rand, 2, 4))
        layer.x = True
        assert layer.y.size() == th.Size([2, 4])

    def test_y_is_undefined_if_gen_and_false_passed(self):
        layer = tako.Layer(Gen(th.rand, 2, 4))
        layer.x = False
        assert layer.y is UNDEFINED

    def test_join_outputs_undefined_if_one_undefined(self):
        x1 = th.rand(2)
        layer = tako.Layer(Null(), x=x1)
        layer2 = tako.Layer(Null())
        layer3 = layer.join(layer2)

        assert layer3.y is UNDEFINED
    

class TestIn:

    def test_x_is_undefined_if_not_set(self):

        in_ = tako.In()
        assert in_.x == tako.UNDEFINED

    def test_y_is_correct_value(self):

        x = th.rand(2)
        in_ = tako.In(x=x)
        assert in_.y is x

    def test_to_is_passed_to_layer(self):

        x = th.rand(2)
        layer = tako.In(x=x).to(nn.Sigmoid())
        assert (layer.y == th.sigmoid(x)).all()

    def test_to_is_undefined_using_to_layer(self):

        layer = tako.In().to(nn.Sigmoid())
        assert layer.y is UNDEFINED

    def test_to_works_with_multiple_modules(self):

        x = th.rand(2)
        layer = tako.In(x).to(nn.Sigmoid()).to(nn.Tanh())
        assert (layer.y == th.tanh(th.sigmoid(x))).all()

    def test_index_works_with_list(self):

        x = [th.rand(2), th.rand(3)]
        layer = tako.In(x)[0]
        assert (x[0] == layer.y).all()


class TestRoute:

    def test_route_with_success_outputting_true(self):
        
        cond = tako.In(True)
        x = th.rand(2)
        success, failure = tako.In(x).route(cond)

        assert (success.y == x).all()
        assert (failure.y == tako.UNDEFINED)      

    def test_route_with_failure_outputting_true(self):
        
        cond = tako.In(False)
        x = th.rand(2)
        success, failure = tako.In(x).route(cond)

        assert (failure.y == x).all()
        assert (success.y == tako.UNDEFINED)        

    def test_route_with_undefined_input(self):
        
        cond = tako.In(False)
        success, failure = tako.In().route(cond)

        assert failure.y is UNDEFINED  
        assert failure.x is UNDEFINED   


class TestFeedback:

    def test_feedback_get_gets_default(self):

        feedback = tako.Feedback('c2', 0)
        result = feedback.get(partial(th.zeros, 2, 2))
        assert (result == th.zeros(2, 2)).all()

    def test_feedback_get_gets_value(self):

        feedback = tako.Feedback('c2', 0)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        result = feedback.get(partial(th.zeros, 2, 2))
        assert (result == x).all()

    def test_feedback_get_gets_value_with_delay(self):

        feedback = tako.Feedback('c2',1)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        feedback.adv()
        result = feedback.get(partial(th.zeros, 2, 2))
        assert (result == x).all()

    def test_feedback_get_gets_default_if_not_set(self):

        feedback = tako.Feedback('c2',0)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        feedback.adv()
        result = feedback.get(partial(th.zeros, 2, 2))
        assert (result == th.zeros(2, 2)).all()

    def test_feedback_in_network(self):

        feedback = tako.Feedback('c2', 0)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        in_ = tako.In(feedback)
        cur_ = tako.cur(in_, partial(th.zeros, 2, 2))
        assert (cur_.y == x).all()

    def test_feedback_in_network_returns_default(self):

        feedback = tako.Feedback('c2', 1)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        in_ = tako.In(feedback)
        cur_ = tako.cur(in_, partial(th.zeros, 2, 2))
        assert (cur_.y == th.zeros(2, 2)).all()

    def test_feedback_retrieval_works_correctly_with_one_delay(self):

        feedback = tako.Feedback('c2', 1)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        in_ = tako.In(feedback)
        cur_ = tako.cur(in_, partial(th.zeros, 2, 2))
        feedback.set(cur_.y, True)
        assert (feedback.get() == x).all()

    def test_updating_feedback_sets_the_feedback(self):

        feedback = tako.Feedback('c2', 1)
        x = th.rand(2, 2)
        feedback.set(x)
        feedback.adv()
        in_ = tako.In(feedback)
        cur_ = tako.cur(in_, partial(th.zeros, 2, 2))
        feedback.set(cur_.y, True)
        feedback.adv()
        assert (feedback.get() == th.zeros(2, 2)).all()


class TestIterator:

    def test_iterator_with_list(self):
        iterator = tako.Iterator([2,3])
        assert iterator.cur == 2
        iterator.adv()
        assert iterator.cur == 3
        iterator.adv()
        assert iterator.is_end() is True

    def test_iterator_end_outputs_undefined(self):
        iterator = tako.Iterator([])
        assert iterator.cur == UNDEFINED
        assert iterator.is_end() is True


def loop_(in_: tako.Node):

    return in_.to(nn.Linear(2, 2))


class TestLoop:

    def test_loop_with_successfully_stacks_modules(self):
        
        iterator = tako.Iterator([th.rand(2),th.rand(2)])
        in_ = tako.In(x=iterator)
        acc, = in_.loop(loop_, tako.All())
        assert acc.y.size() == th.Size([2, 2])

    def test_loop_successfully_chooses_last(self):
        iterator = tako.Iterator([th.rand(2),th.rand(2)])
        in_ = tako.In(x=iterator)
        acc, = in_.loop(loop_, tako.Last())
        assert acc.y.size() == th.Size([2])



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

