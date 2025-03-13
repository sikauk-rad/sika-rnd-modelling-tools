import numpy as np
import polars as pl
import pandas as pd
from beartype import beartype
from numpy.typing import NDArray
from typing import Literal
from collections.abc import Iterable, Sequence
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit, GroupKFold, StratifiedKFold, StratifiedGroupKFold
from xgboost import DMatrix
from numbers import Number
from itertools import chain
from re import sub
from sklearn import metrics
from importlib import import_module
from .types import ArrayType1D2D, ArrayType1D


string_scorers = {
    'R2': metrics.r2_score,
    'MAPE': metrics.mean_absolute_percentage_error,
    'RMSE': lambda y, y_pred: np.sqrt(metrics.mean_squared_error(y,y_pred)),
    'MSE': metrics.mean_squared_error,
    'MAE': metrics.mean_absolute_error,
    'max': metrics.max_error,
    'median': metrics.median_absolute_error,
    # 'accuracy': metrics.accuracy_score,
    # 'balanced_accuracy': metrics.balanced_accuracy_score,

}

string_scorers_literal = Literal[
    'R2',
    'MAPE',
    'MSE',
    'RMSE',
    'max',
    'median',
    'MAE',
]

JSON_INCOMPATIBLE_CHARS = '[^A-Za-z0-9_]+'

def import_class(
    module: str, 
    cls: str | None, 
    name: str,
) -> None:
    mod = getattr(import_module(module), cls) if cls else import_module(module)
    globals()[name] = mod
    return mod


def get_json_incompatible_chars() -> str:
    return JSON_INCOMPATIBLE_CHARS


def check_dimensionality(
    array: np.ndarray | pd.DataFrame | pd.Series | pl.DataFrame,
    ndim: int,
) -> bool:

    return len(array.shape) == ndim


def _handle_X_shape_type(
    X,
) -> tuple[pd.DataFrame | np.ndarray, bool]:

    if not check_dimensionality(X, 2):
        raise ValueError('X must be a 2-dimensional array.')

    elif isinstance(X, pl.DataFrame):
        return (X.to_pandas(), True)

    elif isinstance(X, np.ndarray):
        return (X, False)

    elif isinstance(X, pd.DataFrame):
        return (X, True)

    else:
        raise TypeError('X must be a pandas/polars DataFrame or NumPy array.')



@beartype
def _check_and_get_n_rows(
    *arrays: ArrayType1D2D | None,
    return_iloc: bool = False,
) -> int | tuple[int, list[bool]]:


    # if return_shapes:
    #     shapes, row_counts = [], set()
    #     for array in arrays:
    #         shape = array.shape
    #         row_counts.add(shape[0])
    #         shapes.append(shape)
    if return_iloc:
        pandas, row_counts = [], set()
        for array in arrays:
            if array is not None:
                row_counts.add(array.shape[0])
                pandas.append(isinstance(array, (pd.DataFrame, pd.Series)))
    else:
        row_counts = {array.shape[0] for array in arrays if array is not None}

    if len({*row_counts}) > 1:
        raise ValueError(f'arrays with inhomogeneous row counts {row_counts} found.')
    
    n_rows = next(iter(row_counts))
    return (n_rows, pandas) if return_iloc else n_rows
    # return (n_rows, shapes) if return_shapes else n_rows


@beartype
def general_split(
    *arrays: ArrayType1D2D,
    split_ratio: Sequence[Number],
    stratify: NDArray | None = None,
    groups: NDArray | None = None,
    random_state: int | None = None,
) -> Iterable[ArrayType1D2D]:

    split_ratio = np.array([*split_ratio], dtype = 'float32')
    n_arrays = len(arrays)

    if split_ratio.sum() > 1:
        raise ValueError(
            'sum of split_ratio must be equal to 1 or lower.'
        )
    elif (split_ratio < 0).any() or (split_ratio > 1).any():
        raise ValueError('all split_ratios must exceed 0 and deceed 1.')

    n_rows, is_pds = _check_and_get_n_rows(*arrays, stratify, groups, return_iloc = True)

    train_sizes = split_ratio[:-1].cumsum()
    test_sizes = split_ratio[1:]
    total_sizes = train_sizes + test_sizes
    train_sizes /= total_sizes
    test_sizes /= total_sizes

    train_test_sizes = np.column_stack((train_sizes, test_sizes))[::-1]
    arrays_out = []

    splitter_kwargs = {}
    if groups is not None and stratify is not None:
        raise TypeError('only one of groups and stratify should be provided.')
    elif groups is not None:
        splitter_class = GroupShuffleSplit
        splitter_kw = 'groups'
    elif stratify is not None:
        splitter_class = StratifiedShuffleSplit
        splitter_kw = 'y'
        groups = stratify
    else:
        splitter_class = ShuffleSplit
        splitter_kw = 'groups'

    for train_size, test_size in train_test_sizes:
        try:
            splitter = splitter_class(
                train_size = train_size,
                test_size = test_size,
                random_state = random_state,
            )
            splitter_kwargs[splitter_kw] = groups
            train_idx, test_idx = next(splitter.split(arrays[0], **splitter_kwargs))
        except ValueError:
            splitter = splitter_class(
                train_size = train_size,
                test_size = test_size * 0.99,
                random_state = random_state,
            )
            splitter_kwargs[splitter_kw] = groups
            train_idx, test_idx = next(splitter.split(arrays[0], **splitter_kwargs))

        new_arrays, test_arrays = [], []
        for array, is_pd in zip(arrays, is_pds):
            new_arrays.append(array.iloc[train_idx] if is_pd else array[train_idx])
            test_arrays.append(array.iloc[test_idx] if is_pd else array[test_idx])
        arrays_out.append(test_arrays)
        arrays = new_arrays
        if groups is not None:
            groups = groups[train_idx]

    arrays_out = [*chain(new_arrays, *arrays_out[::-1])]
    rearrayed = []
    for n in range(n_arrays):
        rearrayed.extend(arrays_out[n::n_arrays])

    return rearrayed


class CompatibleGroupKFold(GroupKFold):
    def __init__(
        self,
        groups,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.groups = groups

    def split(
        self,
        *args,
        **kwargs,
    ):
        kwargs.pop('groups', None)
        return super().split(*args, **kwargs, groups = self.groups)


class CompatibleStratifiedKFold(StratifiedKFold):
    def __init__(
        self,
        strata,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.strata = strata

    def split(
        self,
        *args,
        **kwargs,
    ):
        kwargs.pop('groups', None)
        kwargs.pop('y', None)
        return super().split(*args, **kwargs, y = self.strata)


@beartype
def skin_string(string: str) -> str:
    return sub(JSON_INCOMPATIBLE_CHARS, '_', string).strip('_')


@beartype
def DMatrix_to_dataframe(
    dmatrix: DMatrix,
    label: str = 'label',
) -> pd.DataFrame:
    X = pd.DataFrame(
        dmatrix.get_data().toarray(), 
        columns = dmatrix.feature_names,
    )
    X.loc[:,label] = dmatrix.get_label()
    return X


@beartype
def score_from_string(
    y: ArrayType1D,
    y_pred: ArrayType1D,
    scorer: Iterable[string_scorers_literal] | string_scorers_literal,
) -> dict[string_scorers_literal, Number]:

    if isinstance(scorer, str):
        scorer = [scorer]

    scores = {}
    for s in scorer:
        scores[s] = string_scorers[s](y, y_pred)
    return scores


def get_group_means_stds(
    array: NDArray[np.number],
    groups: NDArray,
) -> dict:

    unique_groups, group_idx, group_counts = np.unique(
        groups, 
        return_counts = True, 
        return_inverse = True,
    )
    n_unique_groups = unique_groups.shape[0]
    n_columns = array.shape[1]
    group_idx = group_idx.argsort()

    group_labels = np.repeat(np.arange(n_unique_groups), group_counts)

    group_counts = group_counts[:,None]

    means = np.zeros((n_unique_groups, n_columns), dtype = 'float64')
    means[group_labels] += array[group_idx]
    means /= group_counts

    stds = np.zeros((n_unique_groups, n_columns), dtype = 'float64')
    stds[group_labels] += (array[group_idx] - means[group_labels])**2

    stds /= group_counts
    stds **= 0.5

    return [{
        'group': group,
        'mean': mean,
        'std': std,
        'samples': counts,
    } for group, mean, std, counts in zip(unique_groups, means, stds, group_counts)]