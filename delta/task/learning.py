from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import delta.dataset

from ..dataset import split_dataset
from ..core.strategy import (
    LearningStrategy,
    RandomSelectStrategy,
    EpochMergeStrategy,
    IterMergeStrategy,
    WeightResultStrategy,
    CURVE_TYPE,
    ResultStrategy,
    SelectStrategy,
)
from ..core.task import (
    DataFormat,
    DataLocation,
    DataNode,
    GraphNode,
    InputGraphNode,
    MapOperator,
    MapReduceOperator,
    Operator,
    ReduceOperator,
)
from .task import HorizontalTask

_logger = logging.getLogger(__name__)


class FedAvg(LearningStrategy):
    def __init__(
        self,
        min_clients: int = 2,
        max_clients: int = 2,
        merge_epoch: int = 1,
        merge_iteration: int = 0,
        wait_timeout: float = 60,
        connection_timeout: float = 60,
        precision: int = 8,
        curve: CURVE_TYPE = "secp256k1",
        select_strategy: Type[SelectStrategy] = RandomSelectStrategy,
        result_strategy: Type[ResultStrategy] = WeightResultStrategy,
    ) -> None:
        if merge_epoch > 0 and merge_iteration > 0:
            raise ValueError(
                "merge epoch and merge iteration cannot be both greater than 0"
            )
        if merge_epoch > 0:
            merge_strategy = EpochMergeStrategy(merge_epoch)
        else:
            merge_strategy = IterMergeStrategy(merge_iteration)
        super().__init__(
            "FedAvg",
            select_strategy(min_clients, max_clients),
            merge_strategy,
            result_strategy(),
            wait_timeout,
            connection_timeout,
            False,
            precision,
            curve,
        )


class FaultTolerantFedAvg(LearningStrategy):
    def __init__(
        self,
        min_clients: int = 2,
        max_clients: int = 2,
        merge_epoch: int = 0,
        merge_iteration: int = 0,
        wait_timeout: float = 60,
        connection_timeout: float = 60,
        precision: int = 8,
        curve: CURVE_TYPE = "secp256k1",
        select_strategy: Type[SelectStrategy] = RandomSelectStrategy,
        result_strategy: Type[ResultStrategy] = WeightResultStrategy,
    ) -> None:
        if merge_epoch > 0 and merge_iteration > 0:
            raise ValueError(
                "merge epoch and merge iteration cannot be both greater than 0"
            )
        if merge_epoch <= 0 and merge_iteration <= 0:
            raise ValueError(
                "one of merge epoch and merge iteration should be greater than 0"
            )
        if merge_epoch > 0:
            merge_strategy = EpochMergeStrategy(merge_epoch)
        else:
            merge_strategy = IterMergeStrategy(merge_iteration)

        super().__init__(
            "FaultTolerantFedAvg",
            select_strategy(min_clients, max_clients),
            merge_strategy,
            result_strategy(),
            wait_timeout,
            connection_timeout,
            True,
            precision,
            curve,
        )


class TrainIterator(object):
    def __init__(
        self,
        dataloader: DataLoader,
        epoch: int,
        iteration: int,
        strategy: LearningStrategy,
    ) -> None:
        self.dataloader = dataloader
        self.epoch = epoch
        self.iteration = iteration
        self.strategy = strategy

    def __iter__(self):
        return self._get_iter()

    def _get_iter(self):
        finished = False

        while not finished:
            for batch in self.dataloader:
                if finished:
                    break

                _logger.info(f"Training epoch {self.epoch} iteration {self.iteration}")

                yield batch

                if self.strategy.should_merge(self.epoch, self.iteration, False):
                    _logger.info(f"iteration {self.iteration}, start to merge")
                    finished = True
                self.iteration += 1

            if self.strategy.should_merge(self.epoch, self.iteration, True):
                _logger.info(f"epoch {self.epoch}, start to merge")
                finished = True
            if (self.iteration - 1) % len(self.dataloader) == 0:
                self.epoch += 1


class ValidateIterator(object):
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader

    def __iter__(self):
        return self._get_iter()

    def _get_iter(self):
        iteration = 1
        for batch in self.dataloader:
            _logger.info(f"Validation iteration {iteration}")
            yield batch
            iteration += 1


class HorizontalLearning(HorizontalTask):
    def __init__(
        self,
        name: str,
        max_rounds: int,
        validate_interval: int = 1,
        validate_frac: float = 0.2,
        strategy: LearningStrategy = FaultTolerantFedAvg(
            merge_epoch=1,
        ),
    ) -> None:
        super().__init__(name=name, strategy=strategy)
        self.strategy = strategy
        self.max_rounds = max_rounds

        self.validate_interval = validate_interval
        self.validate_frac = validate_frac
        self.enable_validate = validate_interval > 0

    @abc.abstractmethod
    def make_train_dataloader(self, dataset: Dataset) -> DataLoader:
        ...

    @abc.abstractmethod
    def make_validate_dataloader(self, dataset: Dataset) -> DataLoader:
        ...

    @abc.abstractmethod
    def train(self, dataloader: Iterable):
        ...

    @abc.abstractmethod
    def validate(self, dataloader: Iterable) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        ...

    @abc.abstractmethod
    def dataset(self) -> delta.dataset.Dataset:
        ...

    def _dataset_nodes(
        self, dataset_node: delta.dataset.Dataset
    ) -> Tuple[GraphNode] | Tuple[GraphNode, GraphNode]:
        # train_dataset = DatasetNode(
        #     name="train_dataset",
        #     location=DataLocation.CLIENT,
        #     dataset=self.dataset,
        #     validate=False,
        #     validate_frac=self.validate_frac,
        #     seed=self.seed,
        # )
        # if self.enable_validate:
        #     validate_dataset = DatasetNode(
        #         name="validate_dataset",
        #         location=DataLocation.CLIENT,
        #         dataset=self.dataset,
        #         validate=True,
        #         validate_frac=self.validate_frac,
        #         seed=self.seed,
        #     )
        #     return train_dataset, validate_dataset
        # else:
        #     return (train_dataset,)
        class _SplitDatasetOp(MapOperator):
            def __init__(
                self,
                name: str,
                inputs: List[DataNode],
                outputs: List[DataNode],
                validate: bool,
                validate_frac: float,
            ) -> None:
                super().__init__(
                    name,
                    inputs,
                    outputs,
                    validate=validate,
                    validate_frac=validate_frac,
                )
                self.validate = validate
                self.validate_frac = validate_frac

            def map(
                self, dataset: Dataset | Tuple[Dataset, Dataset]
            ) -> Dataset | Tuple[Dataset, Dataset]:
                if isinstance(dataset, tuple):
                    train_dataset, val_dataset = dataset
                    if self.validate:
                        return train_dataset, val_dataset
                    else:
                        return train_dataset
                else:
                    if self.validate:
                        train_dataset, val_dataset = split_dataset(
                            dataset, self.validate_frac
                        )
                        return train_dataset, val_dataset
                    else:
                        return dataset

        if self.enable_validate:
            train_dataset = GraphNode(
                name="train_dataset", location=DataLocation.CLIENT
            )
            val_dataset = GraphNode(name="val_dataset", location=DataLocation.CLIENT)
            outputs = (train_dataset, val_dataset)
        else:
            train_dataset = GraphNode(
                name="train_dataset", location=DataLocation.CLIENT
            )
            outputs = (train_dataset,)

        split_op = _SplitDatasetOp(
            name="split_dataset",
            inputs=[dataset_node],
            outputs=list(outputs),
            validate=self.enable_validate,
            validate_frac=self.validate_frac,
        )
        for node in outputs:
            node.src = split_op

        return outputs

    def _dataloader_nodes(
        self,
        train_dataset_node: GraphNode,
        val_dataset_node: Optional[GraphNode] = None,
    ) -> Tuple[GraphNode, ...]:
        train_dataloader_node = GraphNode(
            name=f"train_dataloader", location=DataLocation.CLIENT
        )

        class _DataloaderOp(MapOperator):
            def __init__(
                self,
                name: str,
                inputs: List[DataNode],
                outputs: List[DataNode],
                learning: "HorizontalLearning",
                train: bool,
            ) -> None:
                super().__init__(name, inputs, outputs, learning=learning, train=train)
                self.learning = learning
                self.train = train

            def map(self, dataset: Dataset) -> Iterable:
                if self.train:
                    dataloader = self.learning.make_train_dataloader(dataset)
                else:
                    dataloader = self.learning.make_validate_dataloader(dataset)
                return dataloader

        train_dataloader_op = _DataloaderOp(
            name="make_train_dataloader",
            inputs=[train_dataset_node],
            outputs=[train_dataloader_node],
            learning=self,
            train=True,
        )
        train_dataloader_node.src = train_dataloader_op

        res = [train_dataloader_node]

        if val_dataset_node is not None:
            val_dataloader_node = GraphNode(
                name=f"validate_dataloader", location=DataLocation.CLIENT
            )

            val_dataloader_op = _DataloaderOp(
                name="make_validate_dataloader",
                inputs=[val_dataset_node],
                outputs=[val_dataloader_node],
                learning=self,
                train=False,
            )
            val_dataloader_node.src = val_dataloader_op
            res.append(val_dataloader_node)

        return tuple(res)

    def _train_result_node(
        self,
        train_dataloader_node: GraphNode,
        weight_node: GraphNode,
        epoch_node: GraphNode,
        iteration_node: GraphNode,
        round: int,
    ) -> GraphNode:
        new_weight_node = GraphNode(
            name=f"weight_{round}", location=DataLocation.SERVER
        )
        new_epoch_node = GraphNode(name="epoch", location=DataLocation.CLIENT)
        new_iteration_node = GraphNode(name="iteration", location=DataLocation.CLIENT)

        class _TrainOp(MapReduceOperator):
            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                learning: "HorizontalLearning",
                round: int,
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    learning=learning,
                    round=round,
                )
                self.learning = learning
                self.round = round

            def map(
                self,
                dataloader: DataLoader,
                weight: np.ndarray,
                epoch: int,
                iteration: int,
            ) -> Tuple[Dict[str, np.ndarray], int, int]:
                self.learning.strategy.weight_to_params(
                    weight, self.learning.state_dict()
                )
                _logger.info(f"Round {self.round} training")
                train_iter = TrainIterator(
                    dataloader, epoch, iteration, self.learning.strategy
                )
                self.learning.train(train_iter)
                _logger.info(f"Round {self.round} training complete")
                params = self.learning.state_dict()
                res = self.learning.strategy.params_to_result(params, weight)
                return {"res": res}, train_iter.epoch, train_iter.iteration

            def reduce(
                self, data: Dict[str, np.ndarray], node_count: int, weight: np.ndarray
            ) -> np.ndarray:
                res = data.get("res")
                if res is not None:
                    res = res / node_count
                    params = self.learning.state_dict()
                    self.learning.strategy.weight_to_params(weight, params)
                    self.learning.strategy.result_to_params(res, params)
                    new_weight = self.learning.strategy.params_to_weight(params)
                    return new_weight
                else:
                    raise ValueError("aggregate result missing key 'res'")

        train_op = _TrainOp(
            name="train",
            map_inputs=[train_dataloader_node, weight_node, epoch_node, iteration_node],
            reduce_inputs=[weight_node],
            map_outputs=[new_epoch_node, new_iteration_node],
            reduce_outputs=[new_weight_node],
            learning=self,
            round=round,
        )
        new_weight_node.src = train_op

        return new_weight_node

    def _val_result_node(
        self,
        val_dataloader_node: GraphNode,
        weight_node: GraphNode,
        round: int,
        metrics_node: GraphNode | None = None,
    ) -> GraphNode:
        """
        val_result_node: No use. Adding this node can make the dependency more straight forward,
        and simplify the final ouputs nodes.
        """
        new_metrics_node = GraphNode(
            name=f"metrics_{round}", location=DataLocation.SERVER
        )

        class _ValidateOp(MapReduceOperator):
            priority: int = 1

            def __init__(
                self,
                name: str,
                map_inputs: List[DataNode],
                reduce_inputs: List[DataNode],
                map_outputs: List[DataNode],
                reduce_outputs: List[DataNode],
                learning: "HorizontalLearning",
                round: int,
            ) -> None:
                super().__init__(
                    name,
                    map_inputs,
                    reduce_inputs,
                    map_outputs,
                    reduce_outputs,
                    learning=learning,
                    round=round,
                )
                self.learning = learning
                self.round = round

            def map(
                self, dataloader: DataLoader, weight: np.ndarray
            ) -> Dict[str, np.ndarray]:
                self.learning.strategy.weight_to_params(
                    weight, self.learning.state_dict()
                )
                _logger.info(f"Round {self.round} validating")
                val_iter = ValidateIterator(dataloader)
                metrics = self.learning.validate(val_iter)
                _logger.info(f"Round {self.round} validating complete")
                res = {key: np.array(val) for key, val in metrics.items()}
                return res

            def reduce(
                self, data: Dict[str, np.ndarray], node_count: int, metrics=None
            ) -> Dict[str, np.ndarray | float]:
                res = {}
                for key, val in data.items():
                    tmp = val / node_count
                    try:
                        res[key] = tmp.item()
                    except ValueError:
                        res[key] = tmp
                return res

        val_op = _ValidateOp(
            name="validate",
            map_inputs=[val_dataloader_node, weight_node],
            reduce_inputs=[] if metrics_node is None else [metrics_node],
            map_outputs=[],
            reduce_outputs=[new_metrics_node],
            learning=self,
            round=round,
        )
        new_metrics_node.src = val_op

        return new_metrics_node

    def _result_node(
        self, weight_node: GraphNode, metrics_node: GraphNode | None = None
    ) -> GraphNode:
        result_node = GraphNode(name="result", location=DataLocation.SERVER)

        class _ResultOp(ReduceOperator):
            def __init__(
                self,
                name: str,
                inputs: List[DataNode],
                outputs: List[DataNode],
                learning: "HorizontalLearning",
            ) -> None:
                super().__init__(name, inputs, outputs, learning=learning)
                self.learning = learning

            def reduce(
                self,
                weight: np.ndarray,
                metrics: Dict[str, np.ndarray | float] | None = None,
            ) -> Any:
                self.learning.strategy.weight_to_params(
                    weight, self.learning.state_dict()
                )
                return self.learning.state_dict()

        input_nodes: List[DataNode] = [weight_node]
        if metrics_node is not None:
            input_nodes.append(metrics_node)

        result_op = _ResultOp(
            name="result", inputs=input_nodes, outputs=[result_node], learning=self
        )
        result_node.src = result_op
        return result_node

    def _build_graph(self) -> Tuple[List[delta.dataset.Dataset], List[GraphNode]]:
        dataset_node = self.dataset()
        dataset_node.name = "dataset"
        dataset_nodes = self._dataset_nodes(dataset_node)
        dataloader_nodes = self._dataloader_nodes(*dataset_nodes)

        epoch_node = InputGraphNode(
            name="epoch", location=DataLocation.CLIENT, default=1
        )
        iteration_node = InputGraphNode(
            name="iteration", location=DataLocation.CLIENT, default=1
        )
        weight_arr = self.strategy.params_to_weight(self.state_dict())
        weight_node = InputGraphNode(
            name="weight_0", location=DataLocation.SERVER, default=weight_arr
        )
        metrics_node = None
        inputs = [dataset_node, epoch_node, iteration_node, weight_node]
        for i in range(self.max_rounds):
            train_dataloader_node = dataloader_nodes[0]
            if len(dataloader_nodes) > 1:
                val_dataloader_node = dataloader_nodes[1]
            else:
                val_dataloader_node = None

            weight_node = self._train_result_node(
                train_dataloader_node, weight_node, epoch_node, iteration_node, i + 1
            )
            if (
                val_dataloader_node is not None
                and (i + 1) % self.validate_interval == 0
            ):
                metrics_node = self._val_result_node(
                    val_dataloader_node, weight_node, i + 1, metrics_node
                )

        output_node = self._result_node(weight_node, metrics_node)

        return inputs, [output_node]
