from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.task import DataLocation
from ..pandas import DataFrame as _DataFrame
from .file import load_file


class DataFrame(_DataFrame):
    def __init__(
        self,
        dataset: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="",
            location=DataLocation.CLIENT,
            src=None,
            dataset=dataset,
            default=None,
            **kwargs,
        )


def load_dataframe(filename: str) -> pd.DataFrame:
    res = load_file(filename)
    if isinstance(res, pd.DataFrame):
        return res
    else:
        raise ValueError(f"{filename} is not a dataframe")
