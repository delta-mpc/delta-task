from __future__ import annotations

from typing import List, Literal

from .select import SelectStrategy

CURVE_TYPE = Literal[
    "secp192r1", "secp224r1", "secp256k1", "secp256r1", "secp384r1", "secp521r1"
]


class Strategy(object):
    def __init__(
        self,
        name: str,
        select_strategy: SelectStrategy,
        wait_timeout: float = 60,
        connection_timeout: float = 60,
        fault_tolerant: bool = False,
        precision: int = 8,
        curve: CURVE_TYPE = "secp256k1",
    ) -> None:
        self.name = name
        self.select_strategy = select_strategy
        self.wait_timeout = wait_timeout
        self.connection_timeout = connection_timeout
        self.fault_tolerant = fault_tolerant
        self.precision = precision
        self.curve = curve

    def select(
        self, candidates: List[str], last_partners: List[str] | None = None
    ) -> List[str]:
        return self.select_strategy.select(candidates, last_partners)
