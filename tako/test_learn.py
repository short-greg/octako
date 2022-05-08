import pytest
from .learn import (
    Assistant, AssistantGroup, Chart, 
    IterationNotifier, Lecture, Status, Trainer, 
    Validator, Workshop, ValidationCourse
)
import torch.utils.data as data_utils
import torch

N_ELEMENTS = 64
import uuid

class TestChart:

    def test_add_result_creates_dict_with_result(self):
        chart = Chart()
        validator = str(uuid.uuid4())
        chart.register(validator, "Validator")
        chart.update(validator, 1, 0, 10, {"X": 1.0, "Y": 2.0})
        assert chart.df.iloc[0]["X"] == 1.0
        assert chart.df.iloc[0]["Y"] == 2.0

    def test_raises_exception_if_teacher_not_registered(self):
        chart = Chart()
        validator = str(uuid.uuid4())
        with pytest.raises(RuntimeError):
            chart.update(validator, 1, 0, 10, {"X": 1.0, "Y": 2.0})

    def test_add_result_creates_dict_with_two_results(self):
        chart = Chart()
        validator = str(uuid.uuid4())
        chart.register(validator, "Validator")
        chart.update(validator, 1, 0, 10, {"X": 3.0, "Y": 2.0})
        chart.update(validator, 1, 1, 10, {"X": 4.0, "Y": 5.0})
        assert chart.df.iloc[1]["X"] == 4.0
        assert chart.df.iloc[1]["Y"] == 5.0


class DummyAssistant(Assistant):

    def __init__(self, name="Dummy"):
        super().__init__(name)
        self.assisted = False

    def assist(self, status: Status):
        self.assisted = True


class TestAssistant:

    def test_name_is_correct(self):

        assistant = DummyAssistant("X")
        assert assistant.name == "X"


class TestAssistantGroup:
    
    def test_all_in_group_are_not_assisted(self):

        dummy1 = DummyAssistant()
        dummy2 = DummyAssistant()
        assistant_group = AssistantGroup(
            [dummy1, dummy2]
        )
        assert dummy1.assisted is False
        assert dummy2.assisted is False

    def test_all_in_group_are_assisted(self):

        dummy1 = DummyAssistant()
        dummy2 = DummyAssistant()
        assistant_group = AssistantGroup(
            [dummy1, dummy2]
        )
        assistant_group.assist(Status.IN_PROGRESS)
        assert dummy1.assisted is True
        assert dummy2.assisted is True
    
    def test_assistant_group_works_with_none(self):

        assistant_group = AssistantGroup()
        assistant_group.assist(Status.IN_PROGRESS)


class Learner:

    def learn(self, x, t):
        return {"MSE": 1.0}

    def test(self, x, t):
        return {"MSE": 2.0}


def get_dataset():

    return data_utils.TensorDataset(
        torch.randn(N_ELEMENTS, 2), torch.rand(N_ELEMENTS)
    )


class TestTrainer:
    
    def test_trainer_advances_results(self):

        trainer = Trainer("Training", Chart(), Learner(),  get_dataset(), N_ELEMENTS // 2, True)
        trainer.adv()
        assert trainer.status.is_in_progress

    def test_trainer_status_is_finished_after_three_advances(self):

        trainer = Trainer("Training", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2, True)
        trainer.adv()
        trainer.adv()
        trainer.adv()
        assert trainer.status.is_finished

    def test_trainer_results_are_correct(self):

        chart = Chart()
        trainer = Trainer("Training", chart, Learner(), get_dataset(), N_ELEMENTS // 2, True)
        trainer.adv()
        assert chart.df.iloc[0]["MSE"] == 1.0

    def test_reset_updates_status(self):

        trainer = Trainer("Training",  Chart(), Learner(), get_dataset(), N_ELEMENTS // 2, True)
        trainer.adv()
        trainer.epoch()
        assert trainer.status.is_ready

    def test_reset_resets_the_start_of_the_iterator(self):

        trainer = Trainer("Training", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2, True)
        trainer.adv()
        trainer.epoch()
        trainer.adv()
        trainer.adv()
        assert trainer.status.is_in_progress


class TestValidator:
    
    def test_trainer_advances_results(self):

        validator = Validator("validation",Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        validator.adv()
        assert validator.status.is_in_progress

    def test_trainer_status_is_finished_after_three_advances(self):

        validator = Validator("validation", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        validator.adv()
        validator.adv()
        validator.adv()
        assert validator.status.is_finished

    def test_trainer_results_are_correct(self):

        chart = Chart()
        validator = Validator("validation", chart, Learner(), get_dataset(), N_ELEMENTS // 2)
        validator.adv()
        assert chart.df.iloc[0]["MSE"] == 2.0

    def test_reset_updates_status(self):

        validator = Validator("validation", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        validator.adv()
        validator.epoch()
        assert validator.status.is_ready

    def test_reset_resets_the_start_of_the_iterator(self):

        validator = Validator("validation", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        validator.adv()
        validator.epoch()
        validator.adv()
        validator.adv()
        assert validator.status.is_in_progress


class TestLecturer:
    
    def test_lecturer_calls_all_assistants(self):

        dummy1 = DummyAssistant()
        validator = Validator("validation", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , validator, [dummy1])
        lecture.adv()
        assert dummy1.assisted

    def test_lecturer_in_progress_after_advance(self):

        dummy1 = DummyAssistant()
        validator = Validator("validation",Chart(),  Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" ,validator, [dummy1])
        lecture.adv()
        assert lecture.status.is_in_progress

    def test_lecturer_finished_once_validator_finished(self):

        dummy1 = DummyAssistant()
        validator = Validator("validation", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , validator, [dummy1])
        lecture.adv()
        lecture.adv()
        lecture.adv()
        assert lecture.status.is_finished

    def test_lecturer_in_progress_after_reset(self):

        dummy1 = DummyAssistant()
        validator = Validator("validation", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , validator, [dummy1])
        lecture.adv()
        lecture.epoch()
        lecture.adv()
        lecture.adv()
        assert lecture.status.is_in_progress


class TestWorkshop:
    
    def test_workshop_calls_all_assistants(self):

        dummy1 = DummyAssistant()
        validator = Validator("validation", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , validator, [dummy1])
        workshop = Workshop("Training", [lecture], iterations=1)
        workshop.adv()
        assert dummy1.assisted

    def test_lecturer_in_progress_after_advance(self):

        dummy1 = DummyAssistant()
        validator = Validator("validation", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , validator, [dummy1])
        workshop = Workshop("Training", [lecture], iterations=1)
        workshop.adv()
        assert workshop.status.is_in_progress

    def test_lecturer_finished_once_validator_finished(self):

        dummy1 = DummyAssistant()
        validator = Validator("Validator",Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , validator, [dummy1])
        workshop = Workshop("Training", [lecture], iterations=1)
        workshop.adv()
        workshop.adv()
        workshop.adv()
        assert workshop.status.is_finished

    def test_lecturer_is_in_correct_iteration(self):

        dummy1 = DummyAssistant()
        validator = Validator("validation", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , validator, [dummy1])
        workshop = Workshop("Training", [lecture], iterations=1)
        workshop.adv()
        workshop.adv()
        workshop.adv()
        assert workshop.iteration == 1

    def test_lecturer_in_progress_after_reset(self):

        dummy1 = DummyAssistant()
        validator = Validator("validation", Chart(), Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , validator, [dummy1])
        workshop = Workshop("Training", [lecture], iterations=1)
        workshop.adv()
        workshop.epoch()
        workshop.adv()
        workshop.adv()
        assert workshop.status.is_in_progress


class TestIterationNotifier:

    def test_iteration_notifier_does_not_notify(self):

        chart = Chart()
        
        dummy = DummyAssistant()
        notifier = IterationNotifier("Notifier", [dummy],chart, 2)
        chart.register("x", "xy")
        chart.update("x", 0, 0, 10, {})
        notifier.assist(Status.IN_PROGRESS)
        assert dummy.assisted is True

    def test_iteration_notifier_notifies(self):
        chart = Chart()
        
        dummy = DummyAssistant()
        notifier = IterationNotifier("Notifier", [dummy], chart, 2)
        chart.register("x", "xy")
        chart.update("x", 0, 1, 10, {})
        notifier.assist(Status.IN_PROGRESS)
        assert dummy.assisted is True


class TestTrainerBuilder:
    
    def test_lecturer_in_progress_after_advance(self):

        course = ValidationCourse(
            get_dataset(), get_dataset(), 1, 1, Learner()
        )

        course.run()

#     def test_lecturer_finished_once_validator_finished(self):

#         accessor = get_chart_accessor()
#         workshop = (
#             TrainerBuilder()
#             .teacher(get_dataset())
#             .validator(get_dataset())
#         ).build(Learner())
#         workshop.advance(accessor)
#         workshop.advance(accessor)
#         workshop.advance(accessor)
#         workshop.advance(accessor)
#         workshop.advance(accessor)
#         workshop.advance(accessor)
#         assert workshop.status.is_finished

#     def test_lecturer_in_progres_after_trainer_finished(self):

#         accessor = get_chart_accessor()
#         workshop = (
#             TrainerBuilder()
#             .teacher(get_dataset())
#             .validator(get_dataset())
#         ).build(Learner())
#         workshop.advance(accessor)
#         workshop.advance(accessor)
#         workshop.advance(accessor)
#         assert workshop.status.is_in_progress

#     def test_lecturer_in_progres_after_trainer_finished(self):

#         accessor = get_chart_accessor()
#         workshop = (
#             TrainerBuilder()
#             .teacher(get_dataset())
#             .n_epochs(2)
#         ).build(Learner())
#         workshop.advance(accessor)
#         workshop.advance(accessor)
#         workshop.advance(accessor)
#         assert workshop.status.is_in_progress
