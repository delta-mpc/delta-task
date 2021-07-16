import inspect
import json
from io import BytesIO
from typing import Callable, List, Dict, Any, BinaryIO, Iterable, Optional
from zipfile import ZipFile

import dill
import torch

from ..node import Node


class LearningTask(object):
    def __init__(self, name: str,
                 members: List[str],
                 preprocess: Callable,
                 train_step: Callable,
                 dataloader_cfg: Dict[str, Any],
                 total_epoch: int,
                 secure_level: int = 0,
                 merge_round: int = 0,
                 merge_epoch: int = 1
                 ):
        super(LearningTask, self).__init__()
        self._name = name
        self._members = members

        self._preprocess = preprocess
        self._train_step = train_step

        self._dataloader_cfg = dataloader_cfg

        self._total_epoch = total_epoch

        self._secure_level = secure_level
        self._merge_round = 0
        self._merge_epoch = 0

        if merge_round > 0:
            self._merge_round = merge_round
        elif merge_epoch > 0:
            self._merge_epoch = merge_epoch
        else:
            raise ValueError("merge_round and merge_epoch should not all be zero")

        self._models: Dict[str, torch.jit.ScriptModule] = {}
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}

        self._inspect_train_step()

        self._dataloader: Optional[Iterable] = None
        self._state = {}

        self._init_state()

        self._node: Optional[Node] = None

    def _inspect_train_step(self):
        sig = inspect.signature(self._train_step)

        # register models, losses and optimizers from default values
        for name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                val = param.default
                if isinstance(val, torch.nn.Module):
                    self._models[name] = val
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
        self._state["curr_iters"] = 0
        self._state["curr_epochs"] = 0
        self._state["merge_count"] = 0

    def dumps_cfg(self) -> bytes:
        cfg = {
            "name": self._name,
            "members": self._members,
            "dataloader": self._dataloader_cfg,
            "total_epoch": self._total_epoch,
            "secure_level": self._secure_level,
            "merge_round": self._merge_round,
            "merge_epoch": self._merge_epoch,
        }
        optimizers_cfg = self._optimizers_cfg()

        with BytesIO() as file:
            with ZipFile(file, mode="w") as f:
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

    def dump_cfg(self, file: BinaryIO):
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
                "defaults": optimizer.defaults
            }

        return cfg

    @classmethod
    def load_cfg(cls, file: BinaryIO) -> 'LearningTask':
        with ZipFile(file, mode="r") as zip_f:
            names = zip_f.namelist()
            model_files = [name for name in names if name.startswith("models")]
            optimizer_files = [name for name in names if name.startswith("optimizers")]

            with zip_f.open("cfg", mode="r") as f:
                cfg = json.load(f)

            models = {}
            param_lookup = {}
            for file in model_files:
                model_name = file.split("/")[-1]
                with zip_f.open(file, mode="r") as f:
                    model = torch.jit.load(f)
                    models[model_name] = model
                    for name, param in model.named_parameters():
                        param_lookup[model_name + '.' + name] = param

            optimizers = {}
            for file in optimizer_files:
                opt_name = file.split("/")[-1]
                with zip_f.open(file, mode="r") as f:
                    opt_cfg = json.load(f)
                    opt_type = opt_cfg["type"]
                    param_names = opt_cfg["param_names"]
                    state_dict = opt_cfg["state_dict"]
                    opt_defaults = opt_cfg["defaults"]
                    params = []
                    for param_group in param_names:
                        params.append({"params": [param_lookup[name] for name in param_group]})
                    optimizer = getattr(torch.optim, opt_type)(params, **opt_defaults)
                    optimizer.load_state_dict(state_dict)
                    optimizers[opt_name] = optimizer

            with zip_f.open("preprocess", mode="r") as f:
                preprocess = dill.load(f)

            with zip_f.open("train", mode="r") as f:
                train = dill.load(f)

            obj = LearningTask(
                name=cfg["name"],
                members=cfg["members"],
                preprocess=preprocess,
                train_step=train,
                dataloader_cfg=cfg["dataloader"],
                total_epoch=cfg["total_epoch"],
                secure_level=cfg["secure_level"],
                merge_round=cfg["merge_round"],
                merge_epoch=cfg["merge_epoch"]
            )
            return obj

    @classmethod
    def loads_cfg(cls, data: bytes) -> 'LearningTask':
        with BytesIO(data) as f:
            return cls.load_cfg(f)

    def dumps_state_dict(self) -> bytes:
        models = {name: model.state_dict() for name, model in self._models.items()}
        optimizers = {name: opt.state_dict() for name, opt in self._optimizers.items()}
        state_dict = {
            "models": models,
            "optimizers": optimizers,
        }

        with BytesIO() as f:
            torch.save(state_dict, f)
            return f.getvalue()

    def dump_state_dict(self, file: BinaryIO):
        file.write(self.dumps_state_dict())

    def load_state_dict(self, file: BinaryIO):
        state_dict = torch.load(file)
        models = state_dict["models"]
        optimizers = state_dict["optimizers"]

        for name, state in models.items():
            self._models[name].load_state_dict(state)

        for name, state in optimizers.items():
            self._optimizers[name].load_state_dict(state)

    def loads_state_dict(self, data: bytes):
        with BytesIO(data) as f:
            self.load_state_dict(f)

    def load_state(self, file: BinaryIO):
        self._state = json.load(file)

    def dump_state(self, file: BinaryIO):
        json.dump(self._state, file)

    def set_node(self, node: Node):
        self._node = node

    def _setup_dataloader(self):
        dataloader = self._node.get_dataloader(self._dataloader_cfg, self._preprocess)
        self._dataloader = dataloader

    def _should_merge(self, iteration: int, epoch: int) -> bool:
        if self._merge_round > 0 and iteration % self._merge_round == 0:
            return True
        elif self._merge_epoch > 0 and epoch % self._merge_epoch == 0:
            return True
        return False

    def run(self):
        if self._dataloader is None:
            self._setup_dataloader()

        self._node.load_state(self)

        iteration = self._state["curr_iters"]
        epoch = self._state["curr_epochs"]
        while epoch < self._total_epoch:
            for batch in self._dataloader:
                self._train_step(batch, **self._models, **self._optimizers)
                iteration += 1
                if self._should_merge(iteration, epoch):
                    self._node.merge_weight(self)
                self._state["curr_iters"] = iteration
            epoch += 1
            if self._should_merge(iteration, epoch):
                self._node.merge_weight(self)
            self._state["curr_epochs"] = epoch

