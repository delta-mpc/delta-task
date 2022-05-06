from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List


class SelectStrategy(ABC):
    def __init__(self, min_clients: int, max_clients: int) -> None:
        self.min_clients = min_clients
        self.max_clients = max_clients

    @abstractmethod
    def select(
        self, candidates: List[str], last_partners: List[str] | None = None
    ) -> List[str]:
        ...


class RandomSelectStrategy(SelectStrategy):
    def __init__(self, min_clients: int, max_clients: int) -> None:
        super().__init__(min_clients, max_clients)
        self.max_clients = max_clients

    def select(
        self, candidates: List[str], last_partners: List[str] | None = None
    ) -> List[str]:
        if len(candidates) < self.min_clients:
            raise ValueError("not enough clients in select candidates")
        res = candidates
        if len(res) > self.max_clients:
            res = random.sample(res, k=self.max_clients)
        return res


class SameSelectStrategy(SelectStrategy):
    def __init__(self, min_clients: int, max_clients: int) -> None:
        super().__init__(min_clients, max_clients)
        self.max_clients = max_clients

    def select(
        self, candidates: List[str], last_partners: List[str] | None = None
    ) -> List[str]:
        if len(candidates) < self.min_clients:
            raise ValueError("Not enough clients in select candidates")

        if last_partners is None:
            res = candidates
            if len(res) > self.max_clients:
                res = random.sample(res, k=self.max_clients)
            return res
        else:
            assert (
                self.min_clients <= len(last_partners) <= self.max_clients
            ), "Last partners length is invalid"
            return last_partners
