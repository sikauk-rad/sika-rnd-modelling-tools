from typing import TypeVar
import polars as pl
import pandas as pd
from numpy.typing import NDArray

ArrayType1D2D = TypeVar(
    'ArrayType1D2D',
    NDArray,
    pl.DataFrame,
    pd.DataFrame,
    pl.Series,
    pd.Series,
)

ArrayType1D = TypeVar(
    'ArrayType1D',
    pl.Series,
    pd.Series,
    NDArray,
)

ArrayType2D = TypeVar(
    'ArrayType2D',
    NDArray,
    pl.DataFrame,
    pd.DataFrame,
)