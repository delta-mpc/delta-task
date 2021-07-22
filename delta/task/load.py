from typing import BinaryIO
from zipfile import ZipFile

from .learn import LearningTask
from .task import Task

_tasks = {
    "learn": LearningTask,
}


def load(file: BinaryIO) -> Task:
    with ZipFile(file, mode="r") as f:
        if "type" not in f.namelist():
            raise RuntimeError("cannot recogonize task file type")

        with f.open("type", mode="r") as f:
            task_type = f.read().decode("utf-8")

    if task_type not in _tasks:
        raise KeyError(f"unknown task type {task_type}")

    return _tasks[task_type].load_cfg(file)
