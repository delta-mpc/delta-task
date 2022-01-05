from tempfile import TemporaryFile

import httpx

from delta import serialize
from delta.task import Task


class DeltaNode(object):
    def __init__(self, url: str) -> None:
        self._url = url

    def create_task(self, task: Task) -> int:
        url = f"{self._url}/v1/task"
        with TemporaryFile(mode="w+b") as file:
            serialize.dump_task(file, task)
            file.seek(0)
            resp = httpx.post(url, files={"file": file})
            resp.raise_for_status()
            data = resp.json()
            task_id = data["task_id"]
            return task_id
