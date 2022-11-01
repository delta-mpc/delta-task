from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset

import delta.dataset
from delta.delta_node import DeltaNode
from delta.task.learning import FaultTolerantFedAvg, HorizontalLearning


class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 5, padding=2)
        self.pool1 = torch.nn.AvgPool2d(2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5)
        self.pool2 = torch.nn.AvgPool2d(2, stride=2)
        self.dense1 = torch.nn.Linear(400, 100)
        self.dense2 = torch.nn.Linear(100, 10)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 400)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        return x


def transform_data(data: List[Tuple[Image, str]]):
    """
    Used as the collate_fn of dataloader to preprocess the data.
    Resize, normalize the input mnist image, and the return it as a torch.Tensor.
    """
    xs, ys = [], []
    for x, y in data:
        xs.append(np.array(x).reshape((1, 28, 28)))
        ys.append(int(y))

    imgs = torch.tensor(xs)
    label = torch.tensor(ys)
    imgs = imgs / 255 - 0.5
    return imgs, label


class Example(HorizontalLearning):
    def __init__(self) -> None:
        super().__init__(
            name="example",  # The task name which is used for displaying purpose.
            max_rounds=2,  # The number of total rounds of training. In every round, all the nodes calculate their own partial results, and summit them to the server.
            validate_interval=1,  # The number of rounds after which we calculate a validation score.
            validate_frac=0.1,  # The ratio of samples for validate set in the whole dataset，range in (0,1)
            strategy=FaultTolerantFedAvg(  # Strategy for secure aggregation, now available strategies are FedAvg and FaultTolerantFedAvg, in package delta.task.learning
                min_clients=2,  # Minimum nodes required in each round, must be greater than 2.
                max_clients=3,  # Maximum nodes allowed in each round, must be greater equal than min_clients.
                merge_epoch=1,  # The number of epochs to run before aggregation is performed.
                merge_iteration=0,  # The number of iterations to run before aggregation is performed. One of this and the above number must be 0.
                wait_timeout=90,  # Timeout for calculation.
                connection_timeout=10,  # Wait timeout for each step.
            ),
        )
        self.model = LeNet()
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-3,
            nesterov=True,
        )

    def dataset(self) -> delta.dataset.Dataset:
        """
        Define the dataset for task.
        return: an instance of delta.dataset.Dataset
        """
        return delta.dataset.Dataset(dataset="mnist")

    def make_train_dataloader(self, dataset: Dataset) -> DataLoader:
        """
        Define the training dataloader. You can transform the dataset, do some preprocess to the dataset.
        dataset: training dataset
        return: training dataloader
        """
        return DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, collate_fn=transform_data)  # type: ignore

    def make_validate_dataloader(self, dataset: Dataset) -> DataLoader:
        """
        Define the validation dataloader. You can transform the dataset, do some preprocess to the dataset.
        dataset: validation dataset
        return: validation dataloader
        """
        return DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=transform_data)  # type: ignore

    def train(self, dataloader: Iterable):
        """
        The training step defination.
        dataloader: the dataloader corresponding to the dataset.
        return: None
        """
        for batch in dataloader:
            x, y = batch
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self, dataloader: Iterable) -> Dict[str, Any]:
        """
        Validation method.
        To calculate validation scores on each node after several training steps.
        The result will also go through the secure aggregation before sending back to server.
        dataloader: the dataloader corresponding to the dataset.
        return: Dict[str, float], A dictionary with each key (str) corresponds to a score's name and the value (float) to the score's value.
        """
        total_loss = 0
        count = 0
        ys = []
        y_s = []
        for batch in dataloader:
            x, y = batch
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
            total_loss += loss.item()
            count += 1

            y_ = torch.argmax(y_pred, dim=1)
            y_s.extend(y_.tolist())
            ys.extend(y.tolist())
        avg_loss = total_loss / count
        tp = len([1 for i in range(len(ys)) if ys[i] == y_s[i]])
        precision = tp / len(ys)

        return {"loss": avg_loss, "precision": precision}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        The params that need to train and update.
        Only the params returned by this function will be updated and saved during aggregation.
        return: List[torch.Tensor], The list of model params.
        """
        return self.model.state_dict()


if __name__ == "__main__":
    task = Example().build()
    DELTA_NODE_API = "http://127.0.0.1:6700"

    delta_node = DeltaNode(DELTA_NODE_API)
    task_id = delta_node.create_task(task)
    if delta_node.trace(task_id):
        res = delta_node.get_result(task_id)
        print(res)
    else:
        print("Task error")
