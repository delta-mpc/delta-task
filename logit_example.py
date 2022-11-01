from typing import Any, Tuple

import pandas
import delta.dataset
from delta import DeltaNode
from delta.statsmodel import LogitTask, MNLogitTask


class SpectorLogitTask(LogitTask):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            name="spector_logit",  # The task name which is used for displaying purpose.
            min_clients=2,  # Minimum nodes required in each round, must be greater than 2.
            max_clients=3,  # Maximum nodes allowed in each round, must be greater equal than min_clients.
            wait_timeout=5,  # Timeout for calculation.
            connection_timeout=5,  # Wait timeout for each step.
            verify_timeout=360,  # Timeout for the final zero knownledge verification step
            enable_verify=False  # whether to enable final zero knownledge verification step
        )

    def dataset(self):
        return {
            "data": delta.dataset.DataFrame("spector.csv"),
        }

    def preprocess(self, data: pandas.DataFrame) -> Tuple[Any, Any]:
        names = data.columns

        y_name = names[3]
        y = data[y_name].copy()  # type: ignore
        x = data.drop([y_name], axis=1)
        return x, y


class IrisLogitTask(MNLogitTask):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            name="spector_logit",  # The task name which is used for displaying purpose.
            min_clients=2,  # Minimum nodes required in each round, must be greater than 2.
            max_clients=3,  # Maximum nodes allowed in each round, must be greater equal than min_clients.
            wait_timeout=5,  # Timeout for calculation.
            connection_timeout=5,  # Wait timeout for each step.
        )

    def dataset(self):
        return {"data": delta.dataset.DataFrame("iris.csv")}

    def preprocess(self, data: pandas.DataFrame) -> Tuple[Any, Any]:
        y = data["target"].copy()
        x = data.drop(["target"], axis=1)
        return x, y


if __name__ == "__main__":
    task = SpectorLogitTask().build()
    # task = IrisLogitTask().build()

    DELTA_NODE_API = "http://127.0.0.1:6700"

    delta_node = DeltaNode(DELTA_NODE_API)
    task_id = delta_node.create_task(task)
    if delta_node.trace(task_id):
        res = delta_node.get_result(task_id)
        print(res)
    else:
        print("Task error")
