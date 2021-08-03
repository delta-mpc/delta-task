from operator import mod
from delta.node import DebugNode
from delta.task import load as task_load

if __name__ == "__main__":
    with open("task.cfg", mode="rb") as f:
        task = task_load(f)

    node = DebugNode()
    task.id = 1
    task.run(node)
