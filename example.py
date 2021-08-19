import logging

import numpy as np
import torch

from delta import DeltaNode
from delta.task import LearningTask


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


model = LeNet()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3, nesterov=True
)


def train_step(batch, model=model, loss=loss, optimizer=optimizer):
    x, y = batch
    y_pred = model(x)
    loss_val = loss(y_pred, y)
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()


def preprocess(x: np.ndarray, y: str):
    x /= 255.0
    x *= 2
    x -= 1
    x = x.reshape((1, 28, 28))
    return torch.from_numpy(x), torch.tensor(int(y), dtype=torch.long)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    task = LearningTask(
        name="test",
        dataset="mnist",
        preprocess=preprocess,
        train_step=train_step,
        dataloader={"batch_size": 256, "shuffle": True, "drop_last": True},
        total_epoch=2,
        members=[
            "55ZUVDZ4FIUAR6TFSH6I7CP6XPXWYC5AKMYDRJIIUONGLJ6E",
            "Y7L6HUG5BCXWCXR5ADW2ATHA5PX4KNG74I2CBWULOEFVBWH5",
        ],
        merge_iter=20,
    )
    delta_node = DeltaNode("http://127.0.0.1:6701")
    delta_node.create_task(task)
