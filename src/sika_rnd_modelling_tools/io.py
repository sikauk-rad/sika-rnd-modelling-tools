from xgboost import DMatrix, Booster
from beartype import beartype
from pathlib import Path
import pandas as pd
from lightgbm import Booster as LGBooster
from sklearn.base import BaseEstimator
from pickle import dump, load


@beartype
def save_xgboost_model(
    model: Booster,
    out_path: Path,
    *,
    Xy_train: DMatrix | None = None,
    Xy_valid: DMatrix | None = None,
    Xy_test: DMatrix | None = None,
) -> None:

    model.save_model(out_path.joinpath('model.JSON'))
    if Xy_train:
        Xy_train.save_binary(out_path.joinpath('train.bin'))
    if Xy_valid:
        Xy_valid.save_binary(out_path.joinpath('valid.bin'))
    if Xy_test:
        Xy_test.save_binary(out_path.joinpath('test.bin'))
    return


@beartype
def load_xgboost_model(
    model_path: Path,
    *,
    Xy_train_path: Path | None = None,
    Xy_valid_path: Path | None = None,
    Xy_test_path: Path | None = None,
) -> tuple[Booster, list[DMatrix]]:

    model = Booster(params = {'nthread': -1})
    model.load_model(model_path)
    data_out = []
    if Xy_train_path:
        data_out.append(DMatrix(Xy_train_path))
    if Xy_valid_path:
        data_out.append(DMatrix(Xy_valid_path))
    if Xy_test_path:
        data_out.append(DMatrix(Xy_test_path))
    return model, data_out


@beartype
def save_lightgbm_model(
    model: LGBooster,
    out_path: Path,
    *,
    Xy_train: pd.DataFrame | None = None,
    Xy_valid: pd.DataFrame | None = None,
    Xy_test: pd.DataFrame | None = None,
) -> None:

    model.save_model(out_path.joinpath('model.txt'))
    if Xy_train:
        Xy_train.write_parquet(out_path.joinpath('train.parquet'))
    if Xy_valid:
        Xy_valid.write_parquet(out_path.joinpath('valid.parquet'))
    if Xy_test:
        Xy_test.write_parquet(out_path.joinpath('test.parquet'))
    return


def load_lightgbm_model(
    model_path: Path,
    *,
    Xy_train_path: Path | None = None,
    Xy_valid_path: Path | None = None,
    Xy_test_path: Path | None = None,
) -> tuple[LGBooster, list[pd.DataFrame]]:

    model = LGBooster(model_file = model_path)
    data_out = []
    if Xy_train_path:
        data_out.append(pd.read_parquet(Xy_train_path))
    if Xy_valid_path:
        data_out.append(pd.read_parquet(Xy_valid_path))
    if Xy_test_path:
        data_out.append(pd.read_parquet(Xy_test_path))
    return model, data_out


@beartype
def save_sklearn_model(
    model: BaseEstimator,
    out_path: Path,
    *,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> None:

    dump(model, out_path.joinpath('model.pkl'))
    if X_train:
        X_train.to_parquet(out_path.joinpath('X_train.parquet'))
    if y_train:
        y_train.to_parquet(out_path.joinpath('y_train.parquet'))
    if X_test:
        X_test.to_parquet(out_path.joinpath('X_test.parquet'))
    if y_test:
        y_test.to_parquet(out_path.joinpath('y_test.parquet'))
    return


@beartype
def load_sklearn_model(
    model_path: Path,
    *,
    X_train_path: Path | None = None,
    y_train_path: Path | None = None,
    X_test_path: Path | None = None,
    y_test_path: Path | None = None,
) -> tuple[BaseEstimator, list[tuple[pd.DataFrame, pd.Series]]]:

    model = load(model_path)
    train_out = []
    test_out = []
    if X_train_path:
        train_out.append(pd.read_parquet(X_train_path))
    if y_train_path:
        train_out.append(pd.read_parquet(y_train_path))
    if X_test_path:
        test_out.append(pd.read_parquet(X_test_path))
    if y_test_path:
        test_out.append(pd.read_parquet(y_test_path))
    return model, [(*train_out,), (*test_out,)]
