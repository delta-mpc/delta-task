import json
import requests
from tempfile import TemporaryFile
from zipfile import ZipFile

from .task import Task


class Registry(object):
    def __init__(self, url: str) -> None:
        self._url = url

    def create_task(self, task):
        url = f"{self._url}/task"
        with TemporaryFile() as file:
            with ZipFile(file, mode="w") as zip_f:
                metadata = {
                    "name": task.name,
                    "type": task.type,
                    "secure_level": task.secure_level,
                    "algorithm": task.algorithm,
                    "members": task.members,
                }

                with zip_f.open("metadata", mode="w") as f:
                    f.write(json.dumps(metadata).encode("utf-8"))

                with zip_f.open("cfg", mode="w") as f:
                    task.dump_cfg(f)

                with zip_f.open("weight", mode="w") as f:
                    task.dump_weight(f)

            resp = requests.post(url, files={"file": (f"{task.name}.zip", file)})
            resp.raise_for_status()