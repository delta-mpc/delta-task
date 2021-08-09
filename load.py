from io import BytesIO
from delta.node import DebugNode
from delta.task import load as task_load

if __name__ == "__main__":
    with open("task.cfg", mode="rb") as f:
        task = task_load(f)

    node = DebugNode(1)
    with BytesIO() as f:
        task.dump_weight(f)
        f.seek(0)
        node.upload_result(f)
    task.run(node)
