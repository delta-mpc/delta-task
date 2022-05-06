from .select import SameSelectStrategy
from .strategy import CURVE_TYPE, Strategy


class AnalyticsStrategy(Strategy):
    def __init__(
        self,
        min_clients: int = 2,
        max_clients: int = 2,
        wait_timeout: float = 60,
        connection_timeout: float = 60,
        precision: int = 8,
        curve: CURVE_TYPE = "secp256k1",
    ) -> None:
        super().__init__(
            name="analytics",
            select_strategy=SameSelectStrategy(min_clients, max_clients),
            wait_timeout=wait_timeout,
            connection_timeout=connection_timeout,
            fault_tolerant=False,
            precision=precision,
            curve=curve,
        )
