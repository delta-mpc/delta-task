from typing import Dict, Iterable, List, Tuple, Any, Union

import torch

from delta import DeltaNode
from delta.task import HorizontalTask
from delta.algorithm.horizontal import HorizontalAlgorithm, FedAvg


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

class ExampleTask(HorizontalTask):
    def __init__(self):
        super().__init__(
            name="example",
            dataset="mnist",
            max_rounds=2,
            validate_interval=1,
            validate_frac=0.1,
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

    def preprocess(self, x, y=None):
        x /= 255.0
        x *= 2
        x -= 1
        x = x.reshape((1, 28, 28))
        return torch.from_numpy(x), torch.tensor(int(y), dtype=torch.long)

    def train(self, dataloader: Iterable):
        for batch in dataloader:
            x, y = batch
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self, dataloader: Iterable) -> Dict[str, float]:
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

    def get_params(self) -> List[torch.Tensor]:
        return list(self.model.parameters())

    def algorithm(self) -> HorizontalAlgorithm:
        return FedAvg(
            merge_interval_epoch=0,
            merge_interval_iter=20,
            wait_timeout=60,
            connection_timeout=10,
            min_clients=2,
            max_clients=2,
        )

    def dataloader_config(
        self,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        train_config = {"batch_size": 64, "shuffle": True, "drop_last": True}
        val_config = {"batch_size": 64, "shuffle": False, "drop_last": False}
        return train_config, val_config


DELTA_NODE_API = "http://127.0.0.1:6700"

task = ExampleTask()

delta_node = DeltaNode(DELTA_NODE_API)
delta_node.create_task(task)
