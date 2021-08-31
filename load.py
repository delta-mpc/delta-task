from delta.node import DebugNode
from delta.task import Task

if __name__ == "__main__":
    with open("task.cfg", mode="rb") as f:
        task = Task.load(f)
    node = DebugNode(1)
    task.run(node)
