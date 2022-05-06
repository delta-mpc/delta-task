from typing import Any, Dict, Iterable, List, Tuple

import logging
import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset

from delta.debug import debug
from delta.delta_node import DeltaNode
from delta.task.learning import HorizontalLearning, FaultTolerantFedAvg
import delta.dataset

_logger = logging.getLogger("delta")
datefmt = "%Y-%m-%d %H:%M:%S"
common_fmt = "%(asctime)s.%(msecs)03d - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
common_formatter = logging.Formatter(common_fmt, datefmt)
common_handler = logging.StreamHandler()
common_handler.setFormatter(common_formatter)
_logger.addHandler(common_handler)
_logger.setLevel("INFO")


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
            name="example",
            max_rounds=2,
            validate_interval=1,
            validate_frac=0.1,
            strategy=FaultTolerantFedAvg(
                min_clients=2,
                max_clients=3,
                merge_epoch=1,
                wait_timeout=30,
                connection_timeout=10
            )
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
        return delta.dataset.Dataset(dataset="mnist")

    def make_train_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, collate_fn=transform_data)  # type: ignore

    def make_validate_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=transform_data)  # type: ignore

    def train(self, dataloader: Iterable):
        for batch in dataloader:
            x, y = batch
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self, dataloader: Iterable) -> Dict[str, Any]:
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
        return self.model.state_dict()


if __name__ == "__main__":
    example = Example()
    task = example.build()
    # debug(task)
    DELTA_NODE_API = "http://127.0.0.1:6700"

    delta_node = DeltaNode(DELTA_NODE_API)
    delta_node.create_task(task)
