from .delta_node import DeltaNode
from .node import DebugNode
from .task import Task

__all__ = ["DeltaNode", "debug"]


def debug(task: Task):
    debug_node = DebugNode(1)
    task.run(debug_node)
