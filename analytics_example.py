from typing import Dict

import delta.dataset
from delta import pandas as pd
from delta.delta_node import DeltaNode
from delta.task import HorizontalAnalytics


class WageAvg(HorizontalAnalytics):
    def __init__(self) -> None:
        super().__init__(
            name="wage_avg",  # The task name which is used for displaying purpose.
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
        return {"wages": delta.dataset.DataFrame("wages.csv")}

    def execute(self, wages: pd.DataFrame):
        """
        Implementation of analytics.
        input should be the same with keys of returned dict of dataset method.
        """
        return wages.mean()


if __name__ == "__main__":
    task = WageAvg().build()

    DELTA_NODE_API = "http://127.0.0.1:6700"

    delta_node = DeltaNode(DELTA_NODE_API)
    task_id = delta_node.create_task(task)
    if delta_node.trace(task_id):
        res = delta_node.get_result(task_id)
        print(res)
    else:
        print("Task error")
