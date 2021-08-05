import torch
from torch import nn, optim
import numpy as np


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


model = TestModel()
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)


def preprocess(x: np.ndarray):
    y = int(x[0])
    x = x[1:]
    x = (x - 128) / 255
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train_step(
    batch,
    model: nn.Module = model,
    loss: nn.Module = loss,
    optimizer: optim.Optimizer = optimizer,
):
    x, y = batch
    y_pred = model(x)
    loss_val = loss(y_pred, y)
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
