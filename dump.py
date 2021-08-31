from typing import Dict, Iterable, List
import numpy as np
import torch
from torch import nn, optim

from delta.task import HorizontolTask
from delta.algorithm.horizontal import HorizontalAlgorithm, DefaultAlgorithm


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 200)
        self.layer2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


def preprocess(x: np.ndarray):
    y = int(x[0])
    x = x[1:]
    x = (x - 128) / 255
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class TestTask(HorizontolTask):
    def __init__(
        self,
        name: str,
        dataset: str,
        max_epochs: int,
        validate_interval: int = 1,
        validate_frac: float = 0.1,
    ):
        super().__init__(
            name,
            dataset,
            max_epochs,
            validate_interval=validate_interval,
            validate_frac=validate_frac,
        )
        self.model = TestModel()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=1e-3, momentum=0.9, nesterov=True
        )

    def train(self, dataloader: Iterable):
        for batch in dataloader:
            x, y = batch
            y_pred = self.model(x)
            loss_val = self.loss(y_pred, y)
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

    def validate(self, dataloader: Iterable) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            count = 0
            for batch in dataloader:
                x, y = batch
                y_pred = self.model(x)
                loss_val = self.loss(y_pred, y)
                total_loss += loss_val.item()
                count += 1
        self.model.train()
        return {"loss": total_loss / count}

    def get_params(self) -> List[torch.Tensor]:
        return [p for p in self.model.parameters()]

    def preprocess(self, x, y=None):
        y = int(x[0])
        x = x[1:]
        x = (x - 128) / 255
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


if __name__ == "__main__":
    task = TestTask("test", "mnist.npz", 2)
    with open("task.cfg", mode="wb") as f:
        task.dump(f)
