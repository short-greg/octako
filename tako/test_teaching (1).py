import pytest
from .teaching import (
    Assistant, AssistantGroup, Chart, ChartAccessor, 
    IterationNotifier, Lecture, Status, Trainer, 
    TrainerBuilder, Validator, Workshop
)
import torch.utils.data as data_utils
import torch

N_ELEMENTS = 64


class TestChart:

    def test_add_result_creates_dict_with_result(self):
        chart = Chart()
        chart.add_result("Validator", {}, {"X": 1.0, "Y": 2.0})
        assert chart.df.iloc[0]["X"] == 1.0
        assert chart.df.iloc[0]["Y"] == 2.0

    def test_add_result_creates_dict_with_two_results(self):
        chart = Chart()
        chart.add_result("Validator", {}, {"X": 1.0, "Y": 2.0})
        chart.add_result("Validator", {},{"X": 0.5, "Y": 3.0})
        assert chart.df.iloc[1]["X"] == 0.5
        assert chart.df.iloc[1]["Y"] == 3.0


class TestChartAccessor:

    def test_update_updates_the_iterations(self):

        chart = Chart()
        accessor = chart.child("Teacher", "X", "Iterations", 10)
        accessor.update()
        assert accessor.iteration == 1

    def test_chart_returns_the_chart(self):

        chart = Chart()
        accessor = chart.child("Teacher", "X", "Iterations", 10)
        assert accessor.chart is chart

    def test_n_iterations_returns_the_total_number_of_iterations(self):

        chart = Chart()
        accessor = chart.child("Teacher", "X", "Iterations", 10)
        assert accessor.n_iterations == 10

    def test_results_returns_empty_df_when_no_results(self):

        chart = Chart()
        accessor = chart.child("Teacher", "X", "Iterations", 10)
        assert accessor.results is None

    def test_results_returns_correct_results(self):

        chart = Chart()
        accessor = chart.child("Teacher", "X", "Iterations", 10)
        accessor.add_result("Trainer", {"X": 1.0, "Y": 2.0})
        results = accessor.results
        assert results.iloc[0]["X"] == 1.0
        assert results.iloc[0]["Y"] == 2.0

    def test_results_returns_correct_state_in_results(self):

        chart = Chart()
        accessor = chart.child("Teacher", "X", "Iterations", 10)
        accessor.update()
        accessor.add_result("Trainer",{"X": 1.0, "Y": 2.0})
        results = accessor.results
        assert results.iloc[0]["Teacher"] == "X"
        assert results.iloc[0]["Iterations"] == 1


class DummyAssistant(Assistant):

    def __init__(self, name="Dummy"):
        super().__init__(name)
        self.assisted = False

    def assist(self, progress: ChartAccessor, status: Status):
        self.assisted = True


class TestAssistant:

    def test_name_is_correct(self):

        assistant = Assistant("X")
        assert assistant.name == "X"

def get_chart_accessor():
    
    chart = Chart()
    accessor = chart.child("Teacher", "X", "Iterations", 10)
    return accessor


class TestAssistantGroup:
    
    def test_all_in_group_are_not_assisted(self):

        accessor = get_chart_accessor()
        dummy1 = DummyAssistant()
        dummy2 = DummyAssistant()
        assistant_group = AssistantGroup(
            [dummy1, dummy2]
        )
        assert dummy1.assisted is False
        assert dummy2.assisted is False

    def test_all_in_group_are_assisted(self):

        accessor = get_chart_accessor()
        dummy1 = DummyAssistant()
        dummy2 = DummyAssistant()
        assistant_group = AssistantGroup(
            [dummy1, dummy2]
        )
        assistant_group.assist(accessor, Status.IN_PROGRESS)
        assert dummy1.assisted is True
        assert dummy2.assisted is True
    
    def test_assistant_group_works_with_none(self):

        accessor = get_chart_accessor()
        assistant_group = AssistantGroup()
        assistant_group.assist(accessor, Status.IN_PROGRESS)


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

        accessor = get_chart_accessor()
        trainer = Trainer("Training", Learner(), get_dataset(), N_ELEMENTS // 2, True)
        trainer.advance(accessor)
        assert trainer.status.is_in_progress

    def test_trainer_status_is_finished_after_three_advances(self):

        accessor = get_chart_accessor()
        trainer = Trainer("Training", Learner(), get_dataset(), N_ELEMENTS // 2, True)
        trainer.advance(accessor)
        trainer.advance(accessor)
        trainer.advance(accessor)
        assert trainer.status.is_finished

    def test_trainer_results_are_correct(self):

        accessor = get_chart_accessor()
        trainer = Trainer("Training", Learner(), get_dataset(), N_ELEMENTS // 2, True)
        trainer.advance(accessor)
        assert accessor.results.iloc[0]["MSE"] == 1.0

    def test_reset_updates_status(self):

        accessor = get_chart_accessor()
        trainer = Trainer("Training", Learner(), get_dataset(), N_ELEMENTS // 2, True)
        trainer.advance(accessor)
        trainer.reset()
        assert trainer.status.is_ready

    def test_reset_resets_the_start_of_the_iterator(self):

        accessor = get_chart_accessor()
        trainer = Trainer("Training", Learner(), get_dataset(), N_ELEMENTS // 2, True)
        trainer.advance(accessor)
        trainer.reset()
        trainer.advance(accessor)
        trainer.advance(accessor)
        assert trainer.status.is_in_progress


class TestValidator:
    
    def test_trainer_advances_results(self):

        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        validator.advance(accessor)
        assert validator.status.is_in_progress

    def test_trainer_status_is_finished_after_three_advances(self):

        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        validator.advance(accessor)
        validator.advance(accessor)
        validator.advance(accessor)
        assert validator.status.is_finished

    def test_trainer_results_are_correct(self):

        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        validator.advance(accessor)
        assert accessor.results.iloc[0]["MSE"] == 2.0

    def test_reset_updates_status(self):

        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        validator.advance(accessor)
        validator.reset()
        assert validator.status.is_ready

    def test_reset_resets_the_start_of_the_iterator(self):

        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        validator.advance(accessor)
        validator.reset()
        validator.advance(accessor)
        validator.advance(accessor)
        assert validator.status.is_in_progress


class TestLecturer:
    
    def test_lecturer_calls_all_assistants(self):

        dummy1 = DummyAssistant()
        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , "Iteration", validator, [dummy1])
        lecture.advance(accessor)
        assert dummy1.assisted

    def test_lecturer_in_progress_after_advance(self):

        dummy1 = DummyAssistant()
        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , "Iteration", validator, [dummy1])
        lecture.advance(accessor)
        assert lecture.status.is_in_progress

    def test_lecturer_finished_once_validator_finished(self):

        dummy1 = DummyAssistant()
        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , "Iteration", validator, [dummy1])
        lecture.advance(accessor)
        lecture.advance(accessor)
        lecture.advance(accessor)
        assert lecture.status.is_finished

    def test_lecturer_in_progress_after_reset(self):

        dummy1 = DummyAssistant()
        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , "Iteration", validator, [dummy1])
        lecture.advance(accessor)
        lecture.reset()
        lecture.advance(accessor)
        lecture.advance(accessor)
        assert lecture.status.is_in_progress


class TestWorkshop:
    
    def test_workshop_calls_all_assistants(self):

        dummy1 = DummyAssistant()
        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" ,"Iteration", validator, [dummy1])
        workshop = Workshop("Training", "Manager", "Epoch", [lecture], iterations=1)
        workshop.advance(accessor)
        assert dummy1.assisted

    def test_lecturer_in_progress_after_advance(self):

        dummy1 = DummyAssistant()
        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , "Iteration",validator, [dummy1])
        workshop = Workshop("Training", "Manager", "Epoch",  [lecture], iterations=1)
        workshop.advance(accessor)
        assert workshop.status.is_in_progress

    def test_lecturer_finished_once_validator_finished(self):

        dummy1 = DummyAssistant()
        accessor = get_chart_accessor()
        validator = Validator("Validator", Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , "Iteration",validator, [dummy1])
        workshop = Workshop("Training", "Manager", "Epoch",  [lecture], iterations=1)
        workshop.advance(accessor)
        workshop.advance(accessor)
        workshop.advance(accessor)
        assert workshop.status.is_finished

    def test_lecturer_is_in_correct_iteration(self):

        dummy1 = DummyAssistant()
        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , "Iteration", validator, [dummy1])
        workshop = Workshop("Training", "Manager", "Epoch",  [lecture], iterations=1)
        workshop.advance(accessor)
        workshop.advance(accessor)
        workshop.advance(accessor)
        assert workshop.iteration == 1

    def test_lecturer_in_progress_after_reset(self):

        dummy1 = DummyAssistant()
        accessor = get_chart_accessor()
        validator = Validator("validation", Learner(), get_dataset(), N_ELEMENTS // 2)
        lecture = Lecture("Validation" , "Iteration",validator, [dummy1])
        workshop = Workshop("Training", "Manager", "Epoch",  [lecture], iterations=1)
        workshop.advance(accessor)
        workshop.reset()
        workshop.advance(accessor)
        workshop.advance(accessor)
        assert workshop.status.is_in_progress


class TestIterationNotifier:

    def test_iteration_notifier_does_not_notify(self):

        dummy = DummyAssistant()
        accessor = get_chart_accessor()
        notifier = IterationNotifier("Notifier", [dummy], 2)
        notifier.assist(accessor, Status.IN_PROGRESS)
        assert dummy.assisted is False

    def test_iteration_notifier_notifies(self):

        dummy = DummyAssistant()
        accessor = get_chart_accessor()
        notifier = IterationNotifier("Notifier", [dummy], 2)
        accessor.update()
        notifier.assist(accessor, Status.IN_PROGRESS)
        assert dummy.assisted is False


class TestTrainerBuilder:
    
    def test_lecturer_in_progress_after_advance(self):

        accessor = get_chart_accessor()
        workshop = (
            TrainerBuilder()
            .teacher(get_dataset())
            .validator(get_dataset())
        ).build(Learner())
        workshop.advance(accessor)
        assert workshop.status.is_in_progress

    def test_lecturer_finished_once_validator_finished(self):

        accessor = get_chart_accessor()
        workshop = (
            TrainerBuilder()
            .teacher(get_dataset())
            .validator(get_dataset())
        ).build(Learner())
        workshop.advance(accessor)
        workshop.advance(accessor)
        workshop.advance(accessor)
        workshop.advance(accessor)
        workshop.advance(accessor)
        workshop.advance(accessor)
        assert workshop.status.is_finished

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
