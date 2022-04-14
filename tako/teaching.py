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


class Chart(object):
    
    def __init__(self):
        
        self.df = pd.DataFrame()
        self._progress = dict()
        self._progress_cols: typing.Dict[str, set] = dict()
        self._result_cols: typing.Dict[str, set] = dict()
        self._current = None
        self._children = dict()
    
    def child(self, category: str, name: str, iter_name: str, n_iterations: int=None):
        
        if (category, name) in self._children:
            return self._children[(category, name)]

        accessor = ChartAccessor(category, name, iter_name=iter_name, chart=self, n_iterations=n_iterations)
    
        self._children[(category, name)] = accessor
        return accessor

    def add_result(self, teacher: str, progress: dict, result: dict):
        
        self._current = teacher
        self._progress[teacher] = progress
        self._result_cols[teacher] = set(
            [*self._result_cols.get(teacher, []), *result.keys()]
        )
        self._progress_cols[teacher] = set(
            [*self._progress_cols.get(teacher, []), *progress.keys()]
        )
        data = {
            "Teacher": teacher,
            **progress,
            **result
        }
        cur = pd.DataFrame(data, index=[0])
        self.df = pd.concat([self.df, cur], ignore_index=True)
    
    @property
    def current_teacher(self):
        return self._current

    def results(self, teacher: str=None):
        teacher = teacher if teacher is not None else self._current
        return self.df[self.df["Teacher"] == teacher][self._result_cols[teacher]]
    
    def progress(self, teacher: str=None):
        teacher = teacher if teacher is not None else self._current
        return self._progress[teacher]


class ChartAccessor(object):

    def __init__(
        self, name_category: str, name: str, iter_name: str, chart: Chart, 
        n_iterations: int= None, state: dict=None
    ):
        """initializer

        Args:
            category (str): Name of the teacher category
            name (str): Name of the teacher
            iter_name (str): Name of the iterator
            chart (Chart): 
            n_iterations (int, optional): _description_. Defaults to None.
            state (dict, optional): _description_. Defaults to None.
        """
        self._name_category = name_category
        self._chart = chart
        self._state = state or {}
        self._name = name
        self._iter_name = iter_name
        self._n_iterations = n_iterations
        self._cur_iteration = 0
        self._children = dict()
    
    def update(self):
        self._cur_iteration += 1

    @property
    def local_state(self):
        return {
            self._name_category: self._name,
            self._iter_name: self._cur_iteration,
            f'N_{self._iter_name}': self._n_iterations, 
        }
    
    @property
    def chart(self) -> Chart:
        return self._chart

    @property
    def iteration(self) -> int:
        return self._cur_iteration

    @property
    def n_iterations(self) -> int:
        return self._n_iterations
    
    @property
    def results(self) -> typing.Optional[pd.DataFrame]:
        if self._name_category not in set(self._chart.df.columns):
            return None
        return self._chart.df[self._chart.df[self._name_category] == self._name]

    def child(self, category: str, name: str, iter_name: str=None, n_iterations: int=-1):
        """Create a child progress accessor

        Args:
            category (str): Name of the category
            name (str): Name of the teacher
            iter_name (str): Name of the iteration column
            n_iterations (int, optional): Total number of iterations

        Returns:
            ChartAccesssor: Chart accessor with state 
        """
        if (category, name) in self._children:
            return self._children[(category, name)]

        state = {
            **self._state,
            **self.local_state
        }
        accessor = ChartAccessor(
            category, name, iter_name, self._chart, n_iterations, state
        )
        self._children[(category, name)] = accessor
        return accessor

    def add_result(self, teacher: str, result: dict):
        """_summary_

        Args:
            result (dict): _description_
        """
        
        progress = {
            **self.local_state,
            **self._state
        }
        self._chart.add_result(
            teacher, progress, result
        )


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

    def assist(self, progress: ChartAccessor, status: Status):
        pass

    @property
    def name(self):
        return self._name


class AssistantGroup(object):

    def __init__(self, assistants: typing.List[Assistant]=None):

        self._assistants = assistants or []

    def assist(self, progress: ChartAccessor, status: Status):

        for assistant in self._assistants:
            assistant.assist(progress, status)


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
    def suspend(self, progress: ChartAccessor) -> Status:
        pass

    @abstractmethod
    def advance(self, progress: typing.Union[Chart, ChartAccessor]) -> Status:
        pass
        
    def reset(self):
        self._status = Status.READY

    @abstractproperty
    def n_iterations(self) -> int:
        pass


class Teacher(ABC):

    def __init__(self, name: str):
        self._status = Status.READY
        self._name = name

    @abstractmethod
    def advance(self, progress: ChartAccessor) -> Status:
        pass
    
    def reset(self):
        self._status = Status.READY

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
        
    def advance(self, progress: ChartAccessor) -> Status:

        if self._status.is_finished or self._dataloader.is_end():
            self._status = Status.FINISHED
            return self._status
        
        x, t = self._dataloader.cur
        result = self._learner.learn(x, t)
        progress.add_result(self._name, result)
        progress.update()
        self._dataloader.adv()
        self._status = Status.IN_PROGRESS
        return self._status

    def reset(self):
        super().reset()
        self._dataloader = DataLoaderIter(data_utils.DataLoader(
            self._dataset, self._batch_size, shuffle=self._shuffle
        ))
    
    @property
    def n_iterations(self) -> int:
        return len(self._dataloader)


class Validator(Teacher):

    def __init__(self, name: str, learner, dataset: data_utils.Dataset, batch_size: int):
        super().__init__(name)
        self._learner = learner
        self._dataset = dataset
        self._batch_size = batch_size
        self._dataloader = DataLoaderIter(data_utils.DataLoader(
            self._dataset, self._batch_size
        ))

    def advance(self, progress: ChartAccessor) -> Status:

        if self._dataloader.is_end():
            self._status = Status.FINISHED
            return self._status
        
        x, t = self._dataloader.cur
        result = self._learner.test(x, t)
        progress.add_result(self._name, result)
        progress.update()
        self._dataloader.adv()
        self._status = Status.IN_PROGRESS
        return self._status

    def reset(self):
        super().reset()
        self._dataloader = DataLoaderIter(data_utils.DataLoader(
            self._dataset, self._batch_size
        ))

    @property
    def n_iterations(self) -> int:
        return len(self._dataloader)


class ProgressBar(Assistant):

    # pass in the teachers to record the progress for
    def __init__(self, n: int=100):

        self._pbar: tqdm = None
        self._n = n
    
    def start(self, progress: ChartAccessor):
        self._pbar = tqdm()
        self._pbar.refresh()
        self._pbar.reset()
        self._pbar.total = progress.n_iterations

    def finish(self):
        self._pbar.close()
    
    def update(self, progress: ChartAccessor):

        if self._pbar is None:
            self.start(progress)
        chart = progress.chart

        self._pbar.update(1)
        self._pbar.refresh()
        results = chart.results()
        n = min(self._n, len(results))
        self._pbar.set_description_str(chart.current_teacher)
        self._pbar.set_postfix({
            **chart.progress(),
            **results.tail(n).mean(axis=0).to_dict(),
        })

    def assist(self, progress: ChartAccessor, status: Status):

        if status.is_finished or status.is_on_hold:
            self.finish()
        elif status.is_in_progress:
            self.update(progress)
        elif status.is_ready:
            self.start(progress)


class Lecture(Lesson):

    def __init__(
        self, category: str, iter_name: str, trainer: Trainer, 
        assistants: typing.List[Assistant]=None
    ):
        super().__init__(category, trainer.name, iter_name, assistants)
        self._trainer = trainer
        self._cur_iteration = 0

    def advance(self, parent_progress: typing.Union[Chart, ChartAccessor]) -> Status:
        
        progress = parent_progress.child(
            self._category, self._name, self._iter_name, self.n_iterations
        )
        if self._status.is_finished:
            return self._status
    
        if self._status.is_ready:
            self._assistants.assist(progress, self._status)
        
        self._status = self._trainer.advance(progress)
        self._assistants.assist(progress, self._status)
        self._cur_iteration += 1
        return self._status

    def suspend(self, progress: ChartAccessor) -> Status:

        self._status = Status.ON_HOLD
        self._assistants.assist(progress, self._status)
        return self._status

    def iteration(self) -> int:
        return self._cur_iteration
    
    @property
    def n_iterations(self):
        return self._trainer.n_iterations

    def reset(self):
        super().reset()
        self._trainer.reset()


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

    def advance(self, parent_progress: typing.Union[ChartAccessor, Chart]) -> Status:
        
        progress = parent_progress.child(
            self._category, self._name, self._iter_name, self._iterations
        )

        if self._status.is_finished:
            return self._status

        if self._status.is_ready:
            self._assistants.assist(progress, self._status)
        
        status = self._lessons[self._cur_lesson].advance(progress)
        self._update_indices(status)
        
        if self._cur_iteration == self._iterations:
            self._status = Status.FINISHED
            self._assistants.assist(progress, self._status)
            return self._status
        
        if self._cur_lesson == len(self._lessons):
            for lesson in self._lessons:
                lesson.reset()
            self._cur_lesson = 0
        
        self._status = Status.IN_PROGRESS
        self._assistants.assist(progress, self._status)
        return self._status

    def suspend(self, progress: ChartAccessor) -> Status:

        self._status = Status.ON_HOLD
        for lesson in self._lessons:
            lesson.suspend(progress)
        self._assistants.assist(progress, self._status)
        return self._status

    @property
    def iteration(self) -> int:
        return self._cur_iteration

    def reset(self):
        super().reset()
        for lesson in self._lessons:
            lesson.reset()
        self._cur_iteration = 0


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
    def to_notify(self, progress: ChartAccessor, status: Status) -> bool:
        raise NotImplementedError

    def assist(self, progress: ChartAccessor, status: Status):

        if self.to_notify(progress, status):
            self._assistants.assist(progress)
    
    def reset(self):
        super().reset()
        self._assistants.reset()


class NotifierF(Notifier):
    
    def __init__(self, name: str, assistants: typing.List[Assistant], notify_f: typing.Callable):
        super().__init__(name, assistants)
        self._notify_f = notify_f
    
    def to_notify(self, progress: ChartAccessor, status: Status) -> bool:
        return self._notify_f(progress, status)


class IterationNotifier(Notifier):
    """
    """

    def __init__(self, name: str, assistants: typing.List[Assistant], frequency: int):

        super().__init__(name, assistants)
        self._frequency = frequency
    
    def to_notify(self, progress: ChartAccessor, status: Status) -> bool:
        return (not status.is_in_progress) or (progress.iteration != 0 and ((progress.iteration) % self._frequency) == 0)


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


class ValidationCourseDirector(CourseDirector):

    def __init__(
        self, training_dataset: data_utils.Dataset, 
        validation_dataset: data_utils.Dataset,
        batch_size: int, n_epochs: int,
        learner
    ):

        self._training_dataset = training_dataset
        self._validation_dataset = validation_dataset
        self.learner = learner
        self._builder = (
            TrainerBuilder()
            .validator(validation_dataset, batch_size)
            .teacher(training_dataset, batch_size)
            .n_epochs(n_epochs)
        )
    
    def run(self) -> Chart:
        workshop = self._builder.build(self._learner)
        chart = Chart()
        
        status = workshop.advance(chart)
        while not status.is_finished:
            status = workshop.advance(chart)
        return chart


class TestingCourseDirector(CourseDirector):

    def __init__(
        self, training_dataset: data_utils.Dataset, 
        testing_dataset: data_utils.Dataset,
        batch_size: int, n_epochs: int,
        learner
    ):
        self._training_dataset = training_dataset
        self._testing_dataset = testing_dataset
        self._learner = learner
        self._builder = (
            TrainerBuilder()
            .tester(training_dataset, batch_size)
            .teacher(training_dataset, batch_size)
            .n_epochs(n_epochs)
        )
    
    def run(self) -> Chart:
        workshop = self._builder.build(self._learner)
        chart = Chart()
        
        status = workshop.advance(chart)
        while not status.is_finished:
            status = workshop.advance(chart)
        return chart
