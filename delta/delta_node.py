from __future__ import annotations

import pickle
import time
from datetime import datetime
from io import BytesIO
from tempfile import TemporaryFile
from typing import Any, Dict, List

import httpx

from . import serialize
from .core.task import Task


def format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


class DeltaNode(object):
    def __init__(self, url: str) -> None:
        self._url = url

    def create_task(self, task: Task) -> int:
        url = f"{self._url}/v1/task"
        with TemporaryFile(mode="w+b") as task_file, TemporaryFile(mode="w+b") as config_file:
            task_config = {
                "name": task.name,
                "dataset": task.dataset,
                "type": task.type,
                "enable_verify": task.enable_verify,
                "options": task.options
            }
            pickle.dump(task_config, config_file)
            config_file.seek(0)

            serialize.dump_task(task_file, task)
            task_file.seek(0)
            files = {
                "file": ("task_file.pkl", task_file, "application/octet-stream"),
                "config": ("task_config_file.pkl", config_file, "application/pickle")
            }

            resp = httpx.post(url, files=files, timeout=None)
            resp.raise_for_status()
            data = resp.json()
            task_id = data["task_id"]
            return task_id

    def wait(self, task_id: int) -> bool:
        url = f"{self._url}/v1/task/metadata"
        while True:
            resp = httpx.get(url, params={"task_id": task_id}, timeout=None)
            resp.raise_for_status()
            metadata = resp.json()
            status: str = metadata["status"]
            enable_verify: bool = metadata["enable_verify"]

            if status == "ERROR":
                return False
            elif enable_verify and status == "CONFIRMED":
                return True
            elif (not enable_verify) and status == "FINISHED":
                return True

            time.sleep(1)

    def trace(self, task_id: int) -> bool:
        metadata_url = f"{self._url}/v1/task/metadata"
        log_url = f"{self._url}/v1/task/logs"
        start = 0

        while True:
            while True:
                resp = httpx.get(
                    log_url, params={"task_id": task_id, "start": start, "limit": 20}, timeout=None
                )
                resp.raise_for_status()
                log_data: List[Dict[str, Any]] = resp.json()

                if len(log_data) == 0:
                    break

                for log in log_data:
                    start = log["id"] + 1
                    created_at: int = log["created_at"]
                    message: str = log["message"]
                    log_message = f"{format_timestamp(created_at / 1000)}  {message}"
                    print(log_message)

                    tx_hash: str | None = log.get("tx_hash", None)
                    if tx_hash is not None:
                        tx_url = f"https://explorer.deltampc.com/tx/{tx_hash}/internal-transactions"
                        print(tx_url)

            resp = httpx.get(metadata_url, params={"task_id": task_id}, timeout=None)
            resp.raise_for_status()
            metadata = resp.json()
            status: str = metadata["status"]
            enable_verify: bool = metadata["enable_verify"]

            if status == "ERROR":
                return False
            elif enable_verify and status == "CONFIRMED":
                return True
            elif (not enable_verify) and status == "FINISHED":
                return True

            time.sleep(1)

    def get_result(self, task_id: int) -> Any:
        url = f"{self._url}/v1/task/result"

        resp = httpx.get(url, params={"task_id": task_id}, timeout=None)
        resp.raise_for_status()

        with BytesIO() as file:
            for chunk in resp.iter_bytes():
                file.write(chunk)

            file.seek(0)
            res = pickle.load(file)
        return res
