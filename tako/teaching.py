from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
import math
from timeit import repeat
from typing import Generic, Iterator, TypeVar
import typing
import pandas as pd
from sklearn import preprocessing
from torch.utils import data as data_utils
from tqdm import tqdm
from enum import Enum
from .learning import Learner

# i like this better
# could be conflicts in naming
# Epoch Teacher Iteration
#       X       0
# Epoch   Epoch/Iter  Epoch/Teacher Epoch/Teacher/Iter Epoch/Teacher/Results (other results)
# <>      1           Trainer       0                    ...
# 
# make sure the name does not have / in it
# this should make it easy to query

# y, weight = MemberOut(ModFactory, ['weight'])
# MemberSet() <- sets the member based on the input
# probably just need these two

# would need to make it so if the necessary data is available
# it does not execute the module
# Shared() <- maybe i don't need this

# Lesson(
#   'Epoch', [Team(trainer, assistants), Team(validator, assistants)]
# )

class Status(Enum):
    
    READY = 0
    IN_PROGRESS = 1
    FINISHED = 2
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
    
    def update(self, teacher: str, epoch: int, iteration: int, n_iterations: int, result: dict):
        
        self._current = teacher
        self._progress[teacher] = Progress(epoch, iteration, n_iterations)
        self._result_cols[teacher].update(result.keys())
        data = {
            'Teacher': teacher,
            'Epoch': epoch,
            'Iteration': iteration,
            'N Iterations': n_iterations,
            **result
        }
        cur = pd.DataFrame(data, index=[0])
        self.df = pd.concat([self.df, cur], ignore_index=True)
    
    def progress(self, teacher: str=None) -> Progress:
        teacher = self.cur if teacher is None else teacher
        return self._progress[teacher]

    @property
    def cur(self) -> str:
        return self._current

    def results(self, teacher: str=None):
        teacher = teacher if teacher is not None else self._current
        df = self.df[self.df["Teacher"] == teacher]
        return self.df[['Teacher', 'Epoch', 'Iteration', 'N Iterations', *self._result_cols[teacher]]]

    # def score(self, teacher: str=None):
    #     teacher = teacher if teacher is not None else self._current
    #     return self._results[teacher].score(
    #         self._progress[teacher].cur(self.df["Teacher"] == teacher)
    #     )
    
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
        raise NotImplementedError
    
    @abstractproperty
    def cur(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def pos(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def is_end(self) -> bool:
        raise NotImplementedError


class DataLoaderIter(DatasetIterator):

    def __init__(self, dataloader: data_utils.DataLoader):
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


class Assistant(object):

    def __init__(self, name: str):
        self._name = name

    def assist(self, chart: Chart, status: Status):
        pass

    @property
    def name(self):
        return self._name

    def load_state_dict(self, state_dict):
        self._name = state_dict['name']

    def state_dict(self):
        return {
            'name': self._name
        }


class AssistantGroup(object):

    def __init__(self, assistants: typing.List[Assistant]=None):

        self._assistants = assistants or []

    def assist(self, chart: Chart, status: Status):

        for assistant in self._assistants:
            assistant.assist(chart, status)

    def load_state_dict(self, state_dict):
        
        for assistant in self._assistants:
            assistant.load_state_dict(state_dict[assistant.name])

    def state_dict(self):
        
        return {
            assistant.name: assistant.state_dict()
            for assistant in self._assistants
        }


class Lesson(ABC):

    def __init__(self, category: str, name: str, iter_name: str, assistants: typing.List[Assistant]=None):
        super().__init__()
        self._assistants = AssistantGroup(assistants)
        self._category = category
        self._iter_name = iter_name
        self._name = name
        self._status = Status.READY

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> Status:
        return self._status
    
    @abstractmethod
    def suspend(self, chart: Chart) -> Status:
        pass

    @abstractmethod
    def adv(self, chart: Chart) -> Status:
        pass
        
    def epoch(self):
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

    def __init__(self, name: str):
        self._status = Status.READY
        self._name = name

    @abstractmethod
    def adv(self, chart: Chart) -> Status:
        pass
    
    def epoch(self):
        self._status = Status.READY
        self._epoch += 1

    def suspend(self):
        pass

    @property
    def status(self) -> Status:
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

    def __init__(self, name:str, learner, dataset: data_utils.Dataset, batch_size: int, shuffle: bool=True):
        
        super().__init__(name)
        self._learner = learner
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._dataloader = DataLoaderIter(data_utils.DataLoader(
            self._dataset, self._batch_size, shuffle=shuffle
        ))
        
    def adv(self, chart: Chart) -> Status:

        if self._status.is_finished or self._dataloader.is_end():
            self._status = Status.FINISHED
            return self._status
        
        x, t = self._dataloader.cur
        result = self._learner.learn(x, t)
        
        chart.update(
            teacher=self._name,
            epoch=self._epoch,
            iteration=self._iteration,
            n_iterations=self.n_iterations,
            results=result
        )
        # progress.add_result(self._name, result)
        # progress.update()
        self._dataloader.adv()
        self._status = Status.IN_PROGRESS
        return self._status

    def epoch(self):
        super().epoch()
        self._dataloader = DataLoaderIter(data_utils.DataLoader(
            self._dataset, self._batch_size, shuffle=self._shuffle
        ))
    
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

    def __init__(self, name: str, learner, dataset: data_utils.Dataset, batch_size: int):
        super().__init__(name)
        self._learner = learner
        self._dataset = dataset
        self._batch_size = batch_size
        self._dataloader = DataLoaderIter(data_utils.DataLoader(
            self._dataset, self._batch_size
        ))

    def adv(self, chart: Chart) -> Status:

        if self._dataloader.is_end():
            self._status = Status.FINISHED
            return self._status
        
        x, t = self._dataloader.cur
        result = self._learner.test(x, t)
        chart.update(
            teacher=self._name,
            epoch=self._epoch,
            iteration=self._iteration,
            n_iterations=self.n_iterations,
            results=result
        )
        self._dataloader.adv()
        self._status = Status.IN_PROGRESS
        self._iterations += 1
        return self._status

    def epoch(self):
        super().epoch()
        self._dataloader = DataLoaderIter(data_utils.DataLoader(
            self._dataset, self._batch_size
        ))

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

    # pass in the teachers to record the progress for
    def __init__(self, n: int=100):

        self._pbar: tqdm = None
        self._n = n
    
    def start(self, chart: Chart):
        self._pbar = tqdm()
        self._pbar.refresh()
        self._pbar.reset()
        self._pbar.total = chart.progress().n_iterations

    def finish(self):
        self._pbar.close()
    
    def update(self, chart: Chart):

        if self._pbar is None:
            self.start(chart)

        self._pbar.update(1)
        self._pbar.refresh()
        results = chart.results()
        n = min(self._n, len(results))
        self._pbar.set_description_str(chart.cur)
        self._pbar.set_postfix({
            **chart.progress(),
            **results.tail(n).mean(axis=0).to_dict(),
        })

    def assist(self, chart: Chart, status: Status):

        if status.is_finished or status.is_on_hold:
            self.finish()
        elif status.is_in_progress:
            self.update(chart)
        elif status.is_ready:
            self.start(chart)


class Lecture(Lesson):

    def __init__(
        self, category: str, iter_name: str, trainer: Trainer, 
        assistants: typing.List[Assistant]=None
    ):
        super().__init__(category, trainer.name, iter_name, assistants)
        self._trainer = trainer
        self._cur_iteration = 0

    def adv(self, chart: Chart) -> Status:
        
        # progress = parent_progress.child(
        #     self._category, self._name, self._iter_name, self.n_iterations
        # )
        if self._status.is_finished:
            return self._status
    
        if self._status.is_ready:
            self._assistants.assist(chart, self._status)
        
        self._status = self._trainer.adv(chart)
        self._assistants.assist(chart, self._status)
        self._cur_iteration += 1
        return self._status

    def suspend(self, chart: Chart) -> Status:

        self._status = Status.ON_HOLD
        self._assistants.assist(chart, self._status)
        return self._status

    def iteration(self) -> int:
        return self._cur_iteration
    
    @property
    def n_iterations(self):
        return self._trainer.n_iterations

    def epoch(self):
        super().epoch()
        self._trainer.epoch()
    
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

    def __init__(self, category: str, name: str, iter_name: str, lessons: typing.List[Lesson], assistants: typing.List[Assistant]=None, iterations: int=1):
        super().__init__(category, name, iter_name, assistants)
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

    def adv(self, chart: Chart) -> Status:
        
        # progress = parent_progress.child(
        #     self._category, self._name, self._iter_name, self._iterations
        # )

        if self._status.is_finished:
            return self._status

        if self._status.is_ready:
            self._assistants.assist(chart, self._status)
        
        status = self._lessons[self._cur_lesson].adv(chart)
        self._update_indices(status)
        
        if self._cur_iteration == self._iterations:
            self._status = Status.FINISHED
            self._assistants.assist(chart, self._status)
            return self._status
        
        if self._cur_lesson == len(self._lessons):
            for lesson in self._lessons:
                lesson.epoch()
            self._cur_lesson = 0
        
        self._status = Status.IN_PROGRESS
        self._assistants.assist(chart, self._status)
        return self._status

    def suspend(self, chart: Chart) -> Status:

        self._status = Status.ON_HOLD
        for lesson in self._lessons:
            lesson.suspend(chart)
        self._assistants.assist(chart, self._status)
        return self._status

    @property
    def iteration(self) -> int:
        return self._cur_iteration

    def epoch(self):
        super().epoch()
        for lesson in self._lessons:
            lesson.epoch()
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
    def to_notify(self, chart: Chart, status: Status) -> bool:
        raise NotImplementedError

    def assist(self, chart: Chart, status: Status):

        if self.to_notify(chart, status):
            self._assistants.assist(Chart)
    
    def reset(self):
        super().reset()
        self._assistants.reset()


class NotifierF(Notifier):
    
    def __init__(self, name: str, assistants: typing.List[Assistant], notify_f: typing.Callable):
        super().__init__(name, assistants)
        self._notify_f = notify_f
    
    def to_notify(self, chart: Chart, status: Status) -> bool:
        return self._notify_f(chart, status)


class IterationNotifier(Notifier):
    """
    """

    def __init__(self, name: str, assistants: typing.List[Assistant], frequency: int):

        super().__init__(name, assistants)
        self._frequency = frequency
    
    def to_notify(self, chart: Chart, status: Status) -> bool:
        return (not status.is_in_progress) or (chart.progress().iteration != 0 and ((chart.progress().iteration) % self._frequency) == 0)


class TrainerBuilder(object):
    """
    """

    def __init__(self):
        self._teacher_params = None
        self._validator_params = None
        self._tester_params = None
        self._n_epochs = 1
    
    def n_epochs(self, n_epochs: int=1):
        self._n_epochs = n_epochs
        return self

    def teacher(self, dataset: data_utils.Dataset, batch_size: int=2**5):
        self._teacher_params = (dataset, batch_size)
        return self

    def validator(self, dataset: data_utils.Dataset, batch_size: int=2**5):
        self._validator_params = (dataset, batch_size)
        return self

    def tester(self, dataset: data_utils.Dataset, batch_size: int=2**5):
        self._tester_params = (dataset, batch_size)
        return self

    def build(self, learner) -> Workshop:

        sub_teachers = []
        if self._teacher_params is not None:
            sub_teachers.append(Lecture("Learning", "Iteration", Trainer("Trainer", learner, *self._teacher_params)))
        if self._validator_params is not None:
            sub_teachers.append(Lecture("Validation", "Iteration", Validator("Validator", learner, *self._validator_params)))
        
        lessons = []
        if sub_teachers:
            lessons.append(Workshop(
                'Teaching', 'Course',  'Epoch', 
                sub_teachers, iterations=self._n_epochs
            ))
        
        if self._tester_params is not None:
            lessons.append(Lecture("Testing", Validator("Tester", learner, *self._tester_params)))
        assistants = [ProgressBar()]
        
        return Workshop('Training', 'Workshop', 'Step', lessons, assistants)


class CourseDirector(ABC):

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


class StandardCourseDirector(CourseDirector):
    
    def __init__(
        self, training_dataset: data_utils.Dataset, 
        batch_size: int, n_epochs: int,
        learner: Learner
    ):
        self._training_dataset = training_dataset
        self._learner = learner
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._workshop = self._build_workshop()
        self._chart = Chart()
    
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
    
    def run(self) -> Chart:
        
        status = self._workshop.adv(self._chart)
        while not status.is_finished:
            status = self._workshop.adv(self._chart)
        return self._chart

    def maximize(self):
        return False


class ValidationCourseDirector(StandardCourseDirector):

    def __init__(
        self, training_dataset: data_utils.Dataset, 
        validation_dataset: data_utils.Dataset,
        batch_size: int, n_epochs: int,
        learner: Learner
    ):
        super().__init__(training_dataset, batch_size, n_epochs, learner)
        self._validation_dataset = validation_dataset

    def _build_workshop(self) -> Workshop:

        return (
            TrainerBuilder()
            .validator('Validator', self._validation_dataset, self._batch_size)
            .teacher('Teacher', self._training_dataset, self._batch_size)
            .n_epochs(self._n_epochs)
        ).build(self._learner)

    def score(self):
        return self._chart.score('Tester', True)


class TestingCourseDirector(StandardCourseDirector):

    def __init__(
        self, training_dataset: data_utils.Dataset, 
        testing_dataset: data_utils.Dataset,
        batch_size: int, n_epochs: int,
        learner: Learner
    ):
        super().__init__(training_dataset, batch_size, n_epochs, learner)
        self._testing_dataset = testing_dataset
    
    def _build_workshop(self) -> Workshop:
        return (
            TrainerBuilder()
            .tester('Tester', self._training_dataset, self._batch_size)
            .teacher('Teacher' ,self._training_dataset, self._batch_size)
            .n_epochs(self._n_epochs)
        ).build(self._learner)

    def score(self):
        return self._chart.score('Tester', True)
