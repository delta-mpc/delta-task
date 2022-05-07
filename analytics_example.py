from delta import pandas as pd
from delta.task import HorizontalAnalytics
from delta.delta_node import DeltaNode
import delta.dataset

from typing import Dict


class Example(HorizontalAnalytics):
    def __init__(self) -> None:
        super().__init__(
            name="example",  # The task name which is used for displaying purpose.
            min_clients=2,  # Minimum nodes required in each round, must be greater than 2.
            max_clients=3,  # Maximum nodes allowed in each round, must be greater equal than min_clients.
            wait_timeout=5,  # Timeout for calculation.
            connection_timeout=5,  # Wait timeout for each step.
        )

    def dataset(self) -> Dict[str, delta.dataset.DataFrame]:
        """
        Define the data used for analytics.
        return: a dict of which key is the dataset name and value is an instance of delta.dataset.DataFrame
        """
        return {
            "df1": delta.dataset.DataFrame("df1.csv"),
            "df2": delta.dataset.DataFrame("df2.csv"),
        }

    def execute(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Implementation of analytics.
        input should be the same with keys of returned dict of dataset method.
        """
        return df1.sum(), df2.sum()


if __name__ == "__main__":
    task = Example().build()

    DELTA_NODE_API = "http://127.0.0.1:6700"

    delta_node = DeltaNode(DELTA_NODE_API)
    delta_node.create_task(task)
