from delta.debug import debug
from delta import pandas as pd
from delta.task import HorizontalAnalytics
from delta.delta_node import DeltaNode
import delta.dataset

import pandas

from typing import Dict


class Example(HorizontalAnalytics):
    def __init__(self) -> None:
        super().__init__(
            name="example",
            min_clients=2,
            max_clients=3,
            wait_timeout=5,
            connection_timeout=5,
        )

    def dataset(self) -> Dict[str, delta.dataset.DataFrame]:
        return {
            "df1": delta.dataset.DataFrame("df1.csv"),
            "df2": delta.dataset.DataFrame("df2.csv")
        }

    def execute(self, df1: pd.DataFrame, df2: pd.DataFrame):
        s1 = df1.sum()
        s2 = df2.sum()
        return (s1 + s2).sum()


if __name__ == "__main__":
    df1 = pandas.DataFrame({'a': [1,2,3], 'b': [0,1,0]})
    df2 = pandas.DataFrame({'a': [1,1,1], 'b': [0,1,2]})
    example = Example()
    task = example.build()

    DELTA_NODE_API = "http://127.0.0.1:6700"

    delta_node = DeltaNode(DELTA_NODE_API)
    delta_node.create_task(task)
