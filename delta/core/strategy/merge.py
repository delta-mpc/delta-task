from __future__ import annotations

from abc import ABC, abstractmethod


class MergeStrategy(ABC):
    @abstractmethod
    def should_merge(self, epoch: int, iteration: int, epoch_finished: bool) -> bool:
        ...


class EpochMergeStrategy(MergeStrategy):
    def __init__(self, epoch: int) -> None:
        assert epoch > 0, "Epoch should be greater than 0"
        self.epoch = epoch

    def should_merge(self, epoch: int, iteration: int, epoch_finished: bool) -> bool:
        return epoch_finished and epoch % self.epoch == 0


class IterMergeStrategy(MergeStrategy):
    def __init__(self, iteration: int) -> None:
        assert iteration > 0, "Iteration should be greater than 0"
        self.iteration = iteration

    def should_merge(self, epoch: int, iteration: int, epoch_finished: bool) -> bool:
        return (not epoch_finished) and iteration % self.iteration == 0
