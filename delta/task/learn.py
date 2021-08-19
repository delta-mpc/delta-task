import inspect
import json
import logging
from io import BytesIO
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, IO
from zipfile import ZipFile
from tempfile import TemporaryFile

import dill
import torch
import numpy as np

from ..node import Node
from .task import Task


class LearningTask(Task):
    def __init__(
        self,
        name: str,
        dataset: str,
        preprocess: Callable,
        train_step: Callable,
        dataloader: Dict[str, Any],
        total_epoch: int,
        members: List[str] = None,
        secure_level: int = 0,
        algorithm: str = "merge_weight",
        merge_iter: int = 0,
        merge_epoch: int = 1,
    ):
        super(LearningTask, self).__init__()
        self._logger = logging.getLogger(__name__)
        self._name = name
        self._dataset = dataset
        self._members = members
        if self._members is None:
            self._members = []

        self._preprocess = preprocess
        self._train_step = train_step

        self._dataloader = dataloader

        self._total_epoch = total_epoch

        self._secure_level = secure_level
        self._algorithm = algorithm
        self._merge_iter = 0
        self._merge_epoch = 0

        if merge_iter > 0:
            self._merge_iter = merge_iter
        elif merge_epoch > 0:
            self._merge_epoch = merge_epoch
        else:
            raise ValueError("merge_round and merge_epoch should not all be zero")

        self._models: Dict[str, torch.jit.ScriptModule] = {}
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}

        self._inspect_train_step()

        self._state = {}

        self._init_state()

    @property
    def name(self):
        return self._name

    @property
    def type(self) -> str:
        return "learn"

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def secure_level(self) -> int:
        return self._secure_level

    @property
    def members(self) -> Optional[List[str]]:
        return self._members

    def _inspect_train_step(self):
        sig = inspect.signature(self._train_step)

        # register models, losses and optimizers from default values
        for name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                val = param.default
                if isinstance(val, torch.nn.Module):
                    self._models[name] = torch.jit.script(val)
                elif isinstance(val, torch.optim.Optimizer):
                    self._optimizers[name] = val
                else:
                    raise RuntimeError(
                        "train_step function only accept torch.nn.Module "
                        "and torch.optim.Optimizer as default args "
                        f"but got {type(val)}"
                    )
        # remove train step's default value to lightweight the pickled function
        self._train_step.__defaults__ = None

    def _init_state(self):
        self._state["iteration"] = 0
        self._state["epoch"] = 0
        self._state["merge_count"] = 0

    def dumps_cfg(self) -> bytes:
        cfg = {
            "name": self._name,
            "dataset": self._dataset,
            "members": self._members,
            "dataloader": self._dataloader,
            "total_epoch": self._total_epoch,
            "secure_level": self._secure_level,
            "algorithm": self._algorithm,
            "merge_iter": self._merge_iter,
            "merge_epoch": self._merge_epoch,
        }
        optimizers_cfg = self._optimizers_cfg()

        with BytesIO() as file:
            with ZipFile(file, mode="w") as f:
                f.writestr("type", self.type.encode("utf-8"))
                f.writestr("cfg", json.dumps(cfg))

                for name, model in self._models.items():
                    with BytesIO() as buffer:
                        torch.jit.save(torch.jit.script(model), buffer)
                        f.writestr("models/" + name, buffer.getvalue())

                for name, opt_cfg in optimizers_cfg.items():
                    f.writestr("optimizers/" + name, json.dumps(opt_cfg))

                with BytesIO() as buffer:
                    dill.dump(self._preprocess, buffer, byref=False, recurse=True)
                    f.writestr("preprocess", buffer.getvalue())

                with BytesIO() as buffer:
                    dill.dump(self._train_step, buffer, byref=False, recurse=True)
                    f.writestr("train", buffer.getvalue())

            return file.getvalue()

    def dump_cfg(self, file: IO[bytes]):
        file.write(self.dumps_cfg())

    def _optimizers_cfg(self) -> Dict:
        cfg = {}

        param_name_lookup = {}
        for model_name, model in self._models.items():
            for param_name, param in model.named_parameters():
                param_name_lookup[id(param)] = model_name + "." + param_name

        for name, optimizer in self._optimizers.items():
            opt_type = optimizer.__class__.__name__
            opt_param_names = []
            for group in optimizer.param_groups:
                param_name_groups = []
                for p in group["params"]:
                    param_name = param_name_lookup[id(p)]
                    param_name_groups.append(param_name)
                opt_param_names.append(param_name_groups)
            cfg[name] = {
                "type": opt_type,
                "param_names": opt_param_names,
                "state_dict": optimizer.state_dict(),
                "defaults": optimizer.defaults,
            }

        return cfg

    @classmethod
    def load_cfg(cls, file: IO[bytes]) -> "LearningTask":
        with ZipFile(file, mode="r") as zip_f:
            names = zip_f.namelist()
            model_files = [name for name in names if name.startswith("models/")]
            optimizer_files = [name for name in names if name.startswith("optimizers/")]

            with zip_f.open("cfg", mode="r") as f:
                cfg = json.load(f)

            models = {}
            param_lookup = {}
            for filename in model_files:
                model_name = filename.split("/")[-1]
                with zip_f.open(filename, mode="r") as f:
                    model = torch.jit.load(f)
                    models[model_name] = model
                    for name, param in model.named_parameters():
                        param_lookup[model_name + "." + name] = param

            optimizers = {}
            for filename in optimizer_files:
                opt_name = filename.split("/")[-1]
                with zip_f.open(filename, mode="r") as f:
                    opt_cfg = json.load(f)
                    opt_type = opt_cfg["type"]
                    param_names = opt_cfg["param_names"]
                    state_dict = opt_cfg["state_dict"]
                    opt_defaults = opt_cfg["defaults"]
                    params = []
                    for param_group in param_names:
                        params.append(
                            {"params": [param_lookup[name] for name in param_group]}
                        )
                    optimizer = getattr(torch.optim, opt_type)(params, **opt_defaults)
                    optimizer.load_state_dict(state_dict)
                    optimizers[opt_name] = optimizer

            with zip_f.open("preprocess", mode="r") as f:
                preprocess = dill.load(f)

            with zip_f.open("train", mode="r") as f:
                train = dill.load(f)

            obj = LearningTask(preprocess=preprocess, train_step=train, **cfg)
            obj._models = models
            obj._optimizers = optimizers
            return obj

    @classmethod
    def loads_cfg(cls, data: bytes) -> "LearningTask":
        with BytesIO(data) as f:
            return cls.load_cfg(f)

    def dumps_weight(self) -> bytes:
        with BytesIO() as f:
            self.dump_weight(f)
            return f.getvalue()

    def dump_weight(self, file: IO[bytes]):
        arrs = []
        for _, model in self._models.items():
            for _, param in model.named_parameters():
                arr = param.cpu().detach().numpy()
                arrs.append(np.ravel(arr))
        weight_arr = np.concatenate(arrs, axis=0)
        np.savez(file, weight_arr)

    def _load_weight_arr(self, weight_arr: np.ndarray):
        assert len(self._models) > 0
        offset = 0
        for _, model in self._models.items():
            for _, param in model.named_parameters():
                shape = list(param.shape)
                size = param.numel()

                param_arr = weight_arr[offset : offset + size]
                offset += size

                p = (
                    torch.from_numpy(param_arr)
                    .to(param.dtype)
                    .to(param.device)
                    .resize_(shape)
                )
                with torch.no_grad():
                    param.copy_(p)

    def load_weight(self, file: IO[bytes]):
        arr_dict = np.load(file)
        weight_arr = arr_dict["arr_0"]  # type: np.ndarray
        self._load_weight_arr(weight_arr)

    def loads_weight(self, data: bytes):
        with BytesIO(data) as f:
            self.load_weight(f)

    def loads_state(self, data: bytes):
        with BytesIO(data) as f:
            self.load_state(f)

    def load_state(self, file: IO[bytes]):
        state_dict = torch.load(file)
        self._state = state_dict["state"]
        for name, model_state_dict in state_dict["models"].items():
            model = self._models[name]
            model.load_state_dict(model_state_dict)
        for name, opt_state_dict in state_dict["optimizers"].items():
            opt = self._optimizers[name]
            opt.load_state_dict(opt_state_dict)

    def dumps_state(self) -> bytes:
        with BytesIO() as f:
            self.dump_state(f)
            return f.getvalue()

    def dump_state(self, file: IO[bytes]):
        state = {"state": self._state}
        state["models"] = {
            name: model.state_dict() for name, model in self._models.items()
        }
        state["optimizers"] = {
            name: opt.state_dict() for name, opt in self._optimizers.items()
        }
        torch.save(state, file)

    def _should_merge(self, epoch_finish: bool) -> bool:
        if (
            not epoch_finish
            and self._merge_iter > 0
            and self._state["iteration"] % self._merge_iter == 0
        ):
            return True
        elif (
            epoch_finish
            and self._merge_epoch > 0
            and self._state["epoch"] % self._merge_epoch == 0
        ):
            return True
        return False

    def _merge_result(self, node: Node):
        # merge and update weight
        with TemporaryFile(mode="w+b") as f:
            self.dump_state(f)
            f.seek(0)
            node.upload_state(f)
        with TemporaryFile(mode="w+b") as f:
            self.dump_weight(f)
            f.seek(0)
            node.upload_result(f)

    def run(self, node: Node):
        try:
            # initial dataloader
            dataloader = node.new_dataloader(
                self._dataset, self._dataloader, self._preprocess
            )
            # initial state
            with TemporaryFile(mode="w+b") as f:
                if node.download_state(f):
                    f.seek(0)
                    self.load_state(f)
            # initial weight
            with TemporaryFile(mode="w+b") as f:
                if node.download_weight(f):
                    f.seek(0)
                    self.load_weight(f)

            self._logger.info(
                f"train start from epoch {self._state['epoch']} iteration {self._state['iteration']}"
            )
            merged = False
            while self._state["epoch"] < self._total_epoch:
                for batch in dataloader:
                    if merged:
                        with TemporaryFile(mode="w+b") as f:
                            if node.download_weight(f):
                                f.seek(0)
                                self.load_weight(f)
                            else:
                                raise ValueError("download weight failed")
                        merged = False

                    self._logger.info(
                        f"epoch {self._state['epoch']} iteration {self._state['iteration']}"
                    )
                    self._train_step(batch, **self._models, **self._optimizers)
                    self._state["iteration"] += 1
                    if self._should_merge(False):
                        self._logger.info(
                            f"iteration {self._state['iteration']}, start merge"
                        )
                        self._merge_result(node)
                        merged = True
                self._state["epoch"] += 1
                self._logger.info(
                    f"epoch: {self._state['epoch']} total epoch: {self._total_epoch}"
                )
                if self._should_merge(True):
                    self._logger.info(f"epoch {self._state['epoch']}, start merge")
                    self._merge_result(node)
                    merged = True
            if not merged:
                self._logger.info(f"training finished, merge unfinished round")
                self._merge_result(node)
                merged = True

            node.finish()
            self._logger.info(f"training finished, total {self._total_epoch}")
        except Exception as e:
            self._logger.exception(e)

    def update(self, result: np.ndarray):
        self._load_weight_arr(result)
