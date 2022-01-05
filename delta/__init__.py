from tempfile import TemporaryDirectory

from .delta_node import DeltaNode
from .node import DebugNode
from .task import Task

__all__ = ["DeltaNode", "debug", "Task"]


def debug(task: Task):
    with TemporaryDirectory() as tmp_dir:
        for round in range(1, 3):
            debug_node = DebugNode("1", round, tmp_dir)
            task.run(debug_node)
