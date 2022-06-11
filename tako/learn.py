from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, asdict
import math
from os import stat
from timeit import repeat
from typing import Generic, Iterator, TypeVar
import typing
import pandas as pd
from sklearn import preprocessing
from torch.utils import data as data_utils
from tqdm import tqdm
from enum import Enum
# from .learning import Learner
import uuid
"""
Basic learning machine classes. 

A learning machine consists of parameters, operations and the interface
Mixins are used to flexibly define the interface for a learning machine
Each Mixin is a "machine component" which defines an interface for the
user to use. Mixins make it easy to reuse components.

class BinaryClassifierLearner(Learner, Tester, Classifier):
  
  def __init__(self, ...):
      # init operations

  def classify(self, x):
      # classify

  def learn(self, x, t):
      # update parameters of network
    
  def test(self, x, t):
      # evaluate the ntwork
    
  def forward(self, x):
      # standard forward method

"""

from dataclasses import dataclass
import typing
import torch
import torch.optim
import torch.nn as nn
from abc import ABC, abstractmethod


def todevice(f):
    """decorator for converting the args to the device of the learner
    """
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return result

    return wrapper


def cpuresult(f):
    """decorator for converting the results to cpu
    """
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return result.cpu()

    return wrapper


def dict_cpuresult(f):
    """decorator for converting the results in dict format to cpu
    """
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return {
            k: t.cpu() for k, t in result.items()
        }
    return wrapper


@dataclass
class Attributes(ABC):
    pass


@dataclass
class LearnerAttributes(object):

    optim: torch.optim.Optimizer


class MachineComponent(nn.Module):
    """Base class for component. Use to build up a Learning Machine
    """

    def __init__(self):
        super().__init__()
        self._device = 'cpu'

    def is_(self, component_cls):
        if isinstance(self, component_cls):
            return True
    
    def set_device(self, device='cpu'):
        self._device = device
        self.to(device)


class Learner(MachineComponent):
    """Update the machine parameters
    """
    VALIDATION = 'validation'
    LOSS = 'loss'

    @abstractmethod
    def learn(self, x, t):
        """Function for learning the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, x, t):
        """Function for evaluating the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError


class Regressor(MachineComponent):
    """Output a real value
    """

    @abstractmethod
    def regress(self, x: torch.Tensor):
        raise NotImplementedError


class Classifier(MachineComponent):
    """Output a categorical value
    """

    @abstractmethod
    def classify(self, x: torch.Tensor):
        raise NotImplementedError


class Status(Enum):
    """Current status of the lesson
    """
    
    # lesson is ready to start
    READY = 0
    # lesson is in progress
    IN_PROGRESS = 1
    # lesson is finished
    FINISHED = 2
    # lesson is on hold/suspended
    ON_HOLD = 3

    @property
    def is_on_hold(self):
        return self == Status.ON_HOLD

    @property
    def is_finished(self):
        return self == Status.FINISHED
    
    @property
    def is_ready(self):
        return self == Status.READY
    
    @property
    def is_in_progress(self):
        return self == Status.IN_PROGRESS


@dataclass
class Progress:

    epoch: int
    n_iterations: int
    iterations: int
    

class Chart(object):
    
    def __init__(self):
        
        self.df = pd.DataFrame()
        self._progress = dict()
        self._result_cols: typing.Dict[str, set] = dict()
        self._current = None
        self._children = dict()
        self._registered = dict()
    
    def update(self, teacher_id: str, epoch: int, iteration: int, n_iterations: int, result: dict):
        """_summary_

        Args:
            teacher_id (str): id of the teacher
            epoch (int): current epoch for the teacher
            iteration (int): current iteration for the teacher
            n_iterations (int): _description_
            result (dict): _description_

        Raises:
            RuntimeError: _description_
        """

        if teacher_id not in self._registered:
            raise RuntimeError(f"Teacher {teacher_id} has not been registered.")
        self._current = teacher_id
        self._progress[teacher_id] = Progress(epoch, n_iterations, iteration)
        self._result_cols[teacher_id].update(result.keys())
        
        updated = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                updated[k] = v.detach().cpu().numpy()
            else:
                updated[k] = result[k]
        data = {
            'Teacher': self._registered[teacher_id],
            'Epoch': epoch,
            'Iteration': iteration,
            'N Iterations': n_iterations,
            **updated
        }
        cur = pd.DataFrame(data, index=[0])
        self.df = pd.concat([self.df, cur], ignore_index=True)
    
    def progress(self, teacher: str=None) -> Progress:
        teacher = self.cur if teacher is None else teacher
        if teacher is None:
            return None
        return self._progress[teacher]

    @property
    def cur(self) -> str:
        return self._current

    def register(self, id: str, name: str):
        if id in self._registered:
            raise RuntimeError(f"Teacher {id} has already been registered.")
        self._registered[id] = name
        self._result_cols[id] = set()

    def results(self, teacher: str=None):
        teacher = teacher if teacher is not None else self._current
        teacher_name = self._registered[teacher]
        df = self.df[self.df["Teacher"] == teacher_name]
        return df[[*self._result_cols[teacher]]]
    
    def state_dict(self):
        return {
            'progress': self._progress,
            'result_cols': self._result_cols,
            'current': self._current,
            'children': self._children,
            'progress_cols': self._progress_cols,
            'df': self.df
        }

    def load_state_dict(self, state_dict):
        self._progress = state_dict['progress']
        self._progress_cols: typing.Dict[str, set] = state_dict['progress_cols']
        self._result_cols: typing.Dict[str, set] = state_dict['result_cols']
        self._current = state_dict['current']
        self._children = state_dict['children']



class DatasetIterator(ABC):
    """For conveniently iterating over a dataset in a behavior tree
    """

    @abstractmethod
    def adv(self):
        """Advance the data iterator
        """
        raise NotImplementedError
    
    @abstractproperty
    def cur(self):
        """Current value
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        """Reset the iterator to the beginning
        """
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns:
            int: Number of items of batches
        """
        raise NotImplementedError

    @abstractmethod
    def pos(self) -> int:
        """
        Returns:
            int: Current position in the 
        """
        raise NotImplementedError
    
    @abstractmethod
    def is_end(self) -> bool:
        """

        Returns:
            bool: is the DataLoader at the end
        """
        raise NotImplementedError


class DataLoaderIter(DatasetIterator):
    """Wraps a DataLoader so iteration is easier in a course
    """

    def __init__(self, dataloader: data_utils.DataLoader):
        """initializer

        Args:
            dataloader (data_utils.DataLoader): DataLoader to iterate over
        """
        self._dataloader = dataloader
        self._cur_iter = iter(self._dataloader)
        self._finished = False
        self._cur = None
        self._is_start = True
        self._pos = 0
        self._iterate()

    def reset(self, dataloader: data_utils.DataLoader=None):

        self._dataloader = dataloader or self._dataloader
        self._cur_iter = iter(self._dataloader)
        self._finished = False
        self._cur = None
        self._is_start = True
        self._pos = 0
        self._iterate()

    def _iterate(self):
        if self._finished:
            raise StopIteration
        try:
            self._cur = next(self._cur_iter)
        except StopIteration:
            self._cur = None
            self._finished = True

    def adv(self):
        self._iterate()
        self._pos += 1
        if self._is_start:
            self._is_start = False
    
    @property
    def cur(self):
        return self._cur
    
    def __len__(self) -> int:
        return len(self._dataloader)
    
    @property
    def pos(self) -> int:
        return self._pos

    def is_end(self) -> bool:
        return self._finished

    def is_start(self):
        return self._is_start


class Assistant(ABC):

    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def assist(self, status: Status):
        pass

    @property
    def name(self):
        """Name of the assistant
        """
        return self._name

    def load_state_dict(self, state_dict):
        self._name = state_dict['name']

    def state_dict(self):
        return {
            'name': self._name
        }


class AssistantGroup(Assistant):
    """Composite assistant. Makes it easier to use multiple assistants
    """

    def __init__(self, assistants: typing.List[Assistant]=None, name: str="Group"):
        """initializer

        Args:
            assistants (typing.List[Assistant], optional): List of assistants. Defaults to None.
        """
        super().__init__(name)

        self._assistants = assistants or []

    def assist(self, status: Status):
        """Run all assistants in the group

        Args:
            status (Status): Current status
        """

        for assistant in self._assistants:
            assistant.assist(status)

    def load_state_dict(self, state_dict):
        
        for assistant in self._assistants:
            assistant.load_state_dict(state_dict[assistant.name])

    def state_dict(self):
        
        return {
            assistant.name: assistant.state_dict()
            for assistant in self._assistants
        }


class Lesson(ABC):

    def __init__(self, name: str, assistants: typing.List[Assistant]=None):
        """initializer

        Args:
            name (str): Name of the lesson
            assistants (typing.List[Assistant], optional): Lesson assistants. Defaults to None.
        """
        super().__init__()
        self._assistants = AssistantGroup(assistants)
        self._name = name
        self._status = Status.READY

    @property
    def name(self) -> str:
        """
        Returns:
            str: Name of the lesson
        """
        return self._name

    @property
    def status(self) -> Status:
        """
        Returns:
            Status: Current status of the lesson
        """
        return self._status
    
    @abstractmethod
    def suspend(self) -> Status:
        pass

    @abstractmethod
    def adv(self) -> Status:
        pass
        
    def adv_epoch(self):
        self._status = Status.READY

    @abstractproperty
    def n_iterations(self) -> int:
        pass

    def load_state_dict(self, state_dict):
        
        self._assistants.load_state_dict(state_dict['assistants'])
        self._status = state_dict['status']
        self._iter_name = state_dict['iter_name']
        self._name = state_dict['name']
        self._category = state_dict['category']

    def state_dict(self):
        return {
            'status': self._status,
            'iter_name': self._iter_name,
            'name': self._name,
            'category': self._category,
            **self._assistants.state_dict()
        }


class Teacher(ABC):

    def __init__(self, name: str, chart: Chart):
        self._id = str(uuid.uuid4())
        self._status = Status.READY
        self._name = name
        self._epoch = 0
        self._chart = chart
        self._iteration = 0
        chart.register(self._id, name)
    
    def id(self):
        return self._id

    @abstractmethod
    def adv(self) -> Status:
        """Advance the teacher by 1
        """
        raise NotImplementedError
    
    def adv_epoch(self):
        """Update epoch for the teacher
        """
        self._status = Status.READY
        self._iteration = 0
        self._epoch += 1

    def suspend(self):
        """Suspend execution of the teacher
        """
        pass

    @property
    def status(self) -> Status:
        """Current status of the teacher
        """
        return self._status
    
    @abstractproperty
    def n_iterations(self) -> int:
        pass

    @property
    def name(self):
        return self._name

    def load_state_dict(self, state_dict):
        self._status = state_dict['status']
        self._name = state_dict['name']

    def state_dict(self):
        return {
            'status': self._status,
            'name': self._name
        }


class Trainer(Teacher):

    def __init__(self, name: str, chart: Chart, learner: Learner, dataset: data_utils.Dataset, batch_size: int, shuffle: bool=True):
        
        super().__init__(name, chart)
        self._learner = learner
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._dataloader = data_utils.DataLoader(
            self._dataset, self._batch_size, shuffle=True
        )
        self._dataloader_iter = DataLoaderIter(self._dataloader)
        self._n_iterations = len(self._dataloader)
        
    def adv(self) -> Status:

        if self._status.is_finished or self._dataloader_iter.is_end():
            self._status = Status.FINISHED
            return self._status
        
        x, t = self._dataloader_iter.cur
        result = self._learner.learn(x, t)
        
        self._chart.update(
            id=self._id,
            epoch=self._epoch,
            iteration=self._iteration,
            n_iterations=self.n_iterations,
            result=result
        )
        self._dataloader_iter.adv()
        self._status = Status.IN_PROGRESS
        self._iteration += 1
        return self._status

    def adv_epoch(self):
        """Advance teh epoch by 1
        """
        super().adv_epoch()
        self._dataloader_iter = DataLoaderIter(self._dataloader)

        # self._dataloader = DataLoaderIter(data_utils.DataLoader(
        #     self._dataset, self._batch_size, shuffle=self._shuffle
        # ))
    
    def score(self):
        return self._chart.df[
            (self._chart.df["Teacher"] == self._name) & 
            (self._chart.df["Epoch"] == self._epoch)
        ][self._learner.VALIDATION].mean()
    
    @property
    def n_iterations(self) -> int:
        return len(self._dataloader)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._batch_size = state_dict['batch_size']
        self._shuffle = state_dict['shuffle']
        self.reset()

    def state_dict(self):

        return {'batch_size': self._batch_size, **super().state_dict()}


class Validator(Teacher):

    def __init__(self, name: str, chart: Chart, learner: Learner, dataset: data_utils.Dataset, batch_size: int):
        super().__init__(name, chart)
        self._learner = learner
        self._dataset = dataset
        self._batch_size = batch_size
        self._dataloader = data_utils.DataLoader(
            self._dataset, self._batch_size
        )
        self._dataloader_iter = DataLoaderIter(self._dataloader)
        self._n_iterations = len(self._dataloader)

    def adv(self) -> Status:

        if self._dataloader_iter.is_end():
            self._status = Status.FINISHED
            return self._status
        
        x, t = self._dataloader_iter.cur
        result = self._learner.test(x, t)
        self._chart.update(
            id=self._id,
            epoch=self._epoch,
            iteration=self._iteration,
            n_iterations=self.n_iterations,
            result=result
        )
        self._dataloader_iter.adv()
        self._status = Status.IN_PROGRESS
        self._iteration += 1
        return self._status

    def adv_epoch(self):
        super().adv_epoch()
        self._dataloader_iter = DataLoaderIter(self._dataloader)

    def score(self):
        return self._chart.df[
            (self._chart.df["Teacher"] == self._name) & 
            (self._chart.df["Epoch"] == self._epoch)
        ][self._learner.VALIDATION].mean()

    @property
    def n_iterations(self) -> int:
        return len(self._dataloader)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._batch_size = state_dict['batch_size']
        self.reset()

    def state_dict(self):
        return {'batch_size': self._batch_size, **super().state_dict()}


class ProgressBar(Assistant):
    """Assistant that displays a progress bar
    """

    def __init__(self, chart: Chart, n: int=100):

        self._pbar: tqdm = None
        self._n = n
        self._chart = chart
    
    def start(self):
        """Load the progress bar
        """
        self._pbar = tqdm()
        self._pbar.refresh()
        self._pbar.reset()
        if self._chart.progress() is not None:
            self._pbar.total = self._chart.progress().n_iterations

    def finish(self):
        self._pbar.close()
    
    def update(self):
        """
        """

        if self._pbar is None:
            self.start(self._chart)

        self._pbar.update(1)
        results = self._chart.results()
        n = min(self._n, len(results))
        self._pbar.set_description_str(self._chart.cur)
        self._pbar.set_postfix({
            **asdict(self._chart.progress()),
            **results.tail(n).mean(axis=0).to_dict(),
        })
        self._pbar.total = self._chart.progress().n_iterations
        self._pbar.refresh()

    def assist(self, status: Status):
        """Assist execution of the progress bar
        """

        if status.is_finished or status.is_on_hold:
            self.finish()
        elif status.is_in_progress:
            self.update()
        elif status.is_ready:
            self.start()


class Lecture(Lesson):
    """Module that executes a trainer and assistants
    """

    def __init__(
        self, name: str, trainer: Trainer, 
        assistants: typing.List[Assistant]=None
    ):
        """initializer

        Args:
            name (str): Name of the lecture
            trainer (Trainer): Trainer to execute
            assistants (typing.List[Assistant], optional): Assistants to execute. Defaults to None.
        """
        super().__init__(name, assistants)
        self._trainer = trainer
        self._cur_iteration = 0

    def adv(self) -> Status:
        """Advance the lecture by 1

        Returns:
            Status: Status after advancing
        """
        if self._status.is_finished:
            return self._status
    
        if self._status.is_ready:
            self._assistants.assist(self._status)
        
        self._status = self._trainer.adv()
        self._assistants.assist(self._status)
        self._cur_iteration += 1
        return self._status

    def suspend(self) -> Status:
        """Suspend the execution fo the lecture

        Returns:
            Status: Status after suspending
        """

        self._status = Status.ON_HOLD
        self._assistants.assist(self._status)
        return self._status

    def iteration(self) -> int:
        return self._cur_iteration
    
    @property
    def n_iterations(self):
        return self._trainer.n_iterations

    def adv_epoch(self):
        super().adv_epoch()
        self._trainer.adv_epoch()
    
    def load_state_dict(self, state_dict):
        
        self._cur_iteration = state_dict['cur_iteration']
        self._trainer = state_dict['trainer']
        super().load_state_dict(state_dict)

    def state_dict(self):
    
        return {
            'trainer': self._trainer.state_dict(),
            'cur_iteration': self._cur_iteration,
            **super().state_dict()
        }


class Workshop(Lesson):
    """
    Workshop executes a series of lessons
    """

    def __init__(
        self, name: str, lessons: typing.List[Lesson], 
        assistants: typing.List[Assistant]=None, 
        iterations: int=1
    ):
        """initializer

        Args:
            name (str): Name of the workshop
            lessons (typing.List[Lesson]): Lessons to run in sequence
            assistants (typing.List[Assistant], optional): . Defaults to None.
            iterations (int, optional): Number of times to repeat the workshop. Defaults to 1.
        """
        super().__init__(name, assistants)
        self._lessons = lessons
        self._iterations = iterations
        self._cur_iteration = 0
        self._cur_lesson = 0
    
    def _update_indices(self, status: Status):

        if status.is_finished:
            self._cur_lesson += 1
        
        if self._cur_lesson == len(self._lessons):
            self._cur_iteration += 1
        
    def n_iterations(self) -> int:
        return self._iterations

    def adv(self) -> Status:
        """Advance the workshop by 1

        Returns:
            Status: Status after advancing
        """

        if self._status.is_finished:
            return self._status

        if self._status.is_ready:
            self._assistants.assist(self._status)
        
        status = self._lessons[self._cur_lesson].adv()
        self._update_indices(status)
        
        if self._cur_iteration == self._iterations:
            self._status = Status.FINISHED
            self._assistants.assist(self._status)
            return self._status
        
        if self._cur_lesson == len(self._lessons):
            for lesson in self._lessons:
                lesson.adv_epoch()
            self._cur_lesson = 0
        
        self._status = Status.IN_PROGRESS
        self._assistants.assist(self._status)
        return self._status

    def suspend(self) -> Status:
        """Suspend the workshop

        Returns:
            Status: Status after suspending
        """

        self._status = Status.ON_HOLD
        for lesson in self._lessons:
            lesson.suspend()
        self._assistants.assist(self._status)
        return self._status

    @property
    def iteration(self) -> int:
        return self._cur_iteration

    def adv_epoch(self):
        super().adv_epoch()
        for lesson in self._lessons:
            lesson.adv_epoch()
        self._cur_iteration = 0
    
    def load_state_dict(self, state_dict):
        
        for lesson in self._lessons:
            lesson.load_state_dict(state_dict[lesson.name])
        self._cur_lesson = state_dict['cur_lesson']
        self._cur_iteration = state_dict['cur_iteration']
        self._iterations = state_dict['iterations']
        super().load_state_dict(state_dict)

    def state_dict(self):
    
        state_dict = {}
        for lesson in self._lessons:
            state_dict[lesson.name] = lesson.state_dict()
        state_dict['cur_lesson'] = self._cur_lesson
        state_dict['cur_iteration'] = self._cur_iteration 
        state_dict['iterations'] = self._iterations
        state_dict.update(
            super().state_dict()
        )
        return state_dict


class Notifier(Assistant):
    """Assistant that 'triggers' another assistant to begine
    """

    def __init__(self, name: str, assistants: typing.List[Assistant]):
        """initializer

        Args:
            assistants (typing.List[Assistant]): Assitants to notify
        """
        super().__init__(name)
        self._assistants = AssistantGroup(assistants)

    @abstractmethod
    def to_notify(self, status: Status) -> bool:
        raise NotImplementedError

    def assist(self, status: Status):

        if self.to_notify(status):
            self._assistants.assist(status)
    
    def reset(self):
        super().reset()
        self._assistants.reset()


class NotifierF(Notifier):
    """
    Notifier that evaluates a function
    """
    
    def __init__(self, name: str, assistants: typing.List[Assistant], notify_f: typing.Callable):
        """
        Args:
            name (str): Name of the notifier
            assistants (typing.List[Assistant]): Assistants to notify 
            notify_f (typing.Callable[[status], bool]): Function to evaluate. Returns a boolean
        """
        super().__init__(name, assistants)
        self._notify_f = notify_f
    
    def to_notify(self, status: Status) -> bool:
        return self._notify_f(status)


class IterationNotifier(Notifier):
    """
    Notify the assistant when the current progress is at a certain point
    """
    def __init__(self, name: str, assistants: typing.List[Assistant], chart: Chart, frequency: int):

        super().__init__(name, assistants)
        self._frequency = frequency
        self._chart = chart
    
    def to_notify(self, status: Status) -> bool:
        return (not status.is_in_progress) or (self._chart.progress().iterations != 0 and ((self._chart.progress().iterations) % self._frequency) == 0)


class Course(ABC):

    @abstractmethod
    def run(self) -> Chart:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def score(self):
        pass

    @property
    def chart(self):
        return self._chart


class StandardCourse(Course):
    """
    Typical course that consists of 
    1. Training a single learner
    2. Managing the chart
    3. Performing a training pass and testing pass 
    """
    
    def _build_workshop(self) -> Workshop:
        raise NotImplementedError
    
    def load_state_dict(self, state_dict):
        self._learner.load_state_dict(state_dict['learner'])
        self._n_epochs = state_dict['n_epochs']
        self._batch_size = state_dict['batch_size']
        self._chart.load_state_dict(state_dict['chart'])
        self._workshop = self._build_workshop()
        self._workshop.load_state_dict(state_dict['workshop'])

    def state_dict(self):
        return {
            'workshop': self._workshop.state_dict(),
            'chart': self._chart.state_dict(),
            'learner': self._learner.state_dict(),
            'n_epochs': self._n_epochs,
            'batch_size': self._batch_size
        }

    @property
    def status(self) -> Status:
        return self._workshop.status
    
    @abstractproperty
    def chart(self):
        pass
    
    def __iter__(self) -> typing.Iterator:
        status = self._workshop.adv()
        while not status.is_finished:
            yield self._workshop.adv()
    
    def run(self) -> Chart:
        
        status = self._workshop.adv()
        while not status.is_finished:
            status = self._workshop.adv()
        return self._chart

    def maximize(self):
        return False


class ValidationCourse(StandardCourse):
    """
    Course for learning hyperparameters - Use before doing the test course
    """

    def __init__(
        self, training_dataset: data_utils.Dataset, 
        validation_dataset: data_utils.Dataset,
        batch_size: int, n_epochs: int,
        learner: Learner
    ):
        super().__init__()

        self._chart = Chart()
        self._learner = learner
        self._n_epochs = n_epochs
        self._training_dataset = training_dataset
        self._validation_dataset = validation_dataset
        self._batch_size = batch_size
        self._trainer = Trainer("Trainer", self._chart, self._learner, self._training_dataset, self._batch_size)
        self._validator = Validator("Validator", self._chart, self._learner, self._validation_dataset, self._batch_size)

        self._workshop = Workshop(
            "Training Workshop",
            lessons=[Lecture("Training", self._trainer), Lecture("Validation", self._validator)], 
            assistants=[ProgressBar(self._chart)], iterations=self._n_epochs
        )
        self._validation_dataset = validation_dataset

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._learner.load_state_dict(state_dict['learner'])
        self._chart.load_state_dict(state_dict['chart'])

    def state_dict(self):
        return {
            **super().state_dict(),
            'chart': self._chart.state_dict(),
            'learner': self._learner.state_dict(),
        }

    @property
    def chart(self):
        return self._chart
    
    @property
    def trainer(self):
        return self._trainer

    @property
    def validator(self):
        return self._validator

    def score(self):
        return self._validator.score()


class TestingCourse(StandardCourse):
    """
    Course for training and then testing a model
    """

    def __init__(
        self, training_dataset: data_utils.Dataset, 
        testing_dataset: data_utils.Dataset,
        batch_size: int, n_epochs: int,
        learner: Learner
    ):
        super().__init__()
        self._learner = learner
        self._n_epochs = n_epochs
        self._chart = Chart()
        self._training_dataset = training_dataset
        self._testing_dataset = testing_dataset
        self._batch_size = batch_size
        self._trainer = Trainer("Trainer", self._chart, self._learner, self._training_dataset, self._batch_size)
        self._tester = Validator("Tester", self._chart, self._learner, self._testing_dataset, self._batch_size)

        training_workshop = Workshop(
            "Training", 
            lessons=[Lecture("Training", self._trainer)], iterations=self._n_epochs
        )
        self._workshop = Workshop(
            "Testing",
            lessons=[training_workshop, Lecture("Testing", self._tester)], 
            assistants=[ProgressBar(self._chart)],
            iterations=1
        )
        self._testing_dataset = testing_dataset

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._learner.load_state_dict(state_dict['learner'])
        self._chart.load_state_dict(state_dict['chart'])

    def state_dict(self):
        return {
            **super().state_dict(),
            'chart': self._chart.state_dict(),
            'learner': self._learner.state_dict(),
        }

    @property
    def chart(self):
        return self._chart

    @property
    def trainer(self):
        return self._trainer

    @property
    def tester(self):
        return self._tester

    def score(self):
        return self._tester.score()
