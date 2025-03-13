import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numbers import Number
from beartype import beartype
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from .utilities import check_dimensionality, _handle_X_shape_type, score_from_string, string_scorers_literal, general_split
from .types import ArrayType2D, ArrayType1D
from collections.abc import Iterator, Iterable
from typing import Self


@beartype
class PolynomialFeaturesNamed(PolynomialFeatures):

    def __init__(
        self,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.feature_names_out_ = None


    def fit_transform(
        self,
        X: ArrayType2D,
        y = None,
        **fit_params,
    ) -> ArrayType2D:

        X, has_columns = _handle_X_shape_type(X)
        X_transform = super().fit_transform(X, y, **fit_params)
        if has_columns:
            self.feature_names_out_ = super().get_feature_names_out()
            return pd.DataFrame(
                X_transform,
                index = X.index,
                columns = self.feature_names_out_,
            )
        else:
            return X_transform


    def transform(
        self,
        X,
    ) -> ArrayType2D:

        X_transform = super().transform(X)
        if self.feature_names_out_ is not None:
            return pd.DataFrame(
                X_transform,
                index = X.index if hasattr(X, 'index') else None,
                columns = self.feature_names_out_,
            )
        else:
            return X_transform



class ColumnSelector(BaseEstimator):

    def __init__(
        self,
        column_idx: list[int] | NDArray[np.integer],
        column_names: list[str] | None = None,
    ) -> None:

        self.column_idx = column_idx
        self.column_names = column_names


    def fit(
        self
    ) -> Self:

        return self


    def transform(
        self,
        X: ArrayType2D,
    ) -> ArrayType2D:

        if isinstance(X, pd.DataFrame):
            if self.column_names is not None:
                return X.loc[:,self.column_names]
            else:
                return X.iloc[:,self.column_idx]
        elif isinstance(X, np.ndarray):
            return X[:,self.column_idx]
        else:
            raise TypeError('X must be a NumPy array or pandas DataFrame.')


    def fit_transform(
        self,
        X: ArrayType2D,
    ) -> ArrayType2D:

        return self.transform(X)


@beartype
class LinearModel(BaseEstimator):

    def __init__(
        self,
        p_threshold: Number | None = 0.05,
        fit_intercept: bool = True,
        polynomial_degree: int = 1,
    ) -> None:

        self.fit_intercept = fit_intercept
        self.polynomial_degree = polynomial_degree
        self._is_fit = False
        if p_threshold is None or p_threshold <= 0:
            self.p_threshold = 0
            self._select_features = False
        else:
            self.p_threshold = p_threshold
            self._select_features = True


    @staticmethod
    def select_features(
        X: NDArray[np.number],
        y: NDArray[np.number],
        p_threshold: Number,
        fit_intercept: bool,
        sample_weights: NDArray[np.number] | int = 1,
    ) -> NDArray[np.bool_]:

        X = sm.add_constant(X) if fit_intercept else X
        model = sm.WLS(
            endog = y,
            exog = X,
            weights = sample_weights,
            missing = 'raise',
        ).fit()
        model_statistics = model.summary2().tables[1]
        p_values = model_statistics.iloc[int(fit_intercept):, 3].values
        p_mask = p_values <= p_threshold
        if not p_mask.any():
            p_threshold = np.nanmin(p_values)
            print(f'p_threshold overridden to {p_threshold}.')
            return p_values <= p_threshold
        else:
            return p_mask


    def fit(
        self,
        X: ArrayType2D,
        y: ArrayType1D,
        sample_weights: NDArray[np.number] | int = 1,
    ) -> Self:

        if not check_dimensionality(y, 1):
            raise ValueError('y must be a 1-dimensional array.')
        X, has_columns = _handle_X_shape_type(X)
        self.feature_names_in_ = X.columns if has_columns else None

        transforms = []

        if self.polynomial_degree > 1:
            polynomial_transformer = PolynomialFeaturesNamed(
                degree = self.polynomial_degree,
                include_bias = False,
            )
            X = polynomial_transformer.fit_transform(X)
            transforms.append(('polynomialise', polynomial_transformer))
            self._feature_names_in = X.columns if has_columns else None
        else:
            self._feature_names_in = self.feature_names_in_

        self._coef = np.zeros(X.shape[1], dtype = 'float64')
        if self._select_features:
            feature_mask = self.select_features(
                X = X.values if has_columns else X,
                y = y.values if has_columns else y,
                p_threshold = self.p_threshold,
                fit_intercept = self.fit_intercept,
                sample_weights = sample_weights,
            )
            self.feature_names_out_ = X.columns[feature_mask] if has_columns else None
            column_selector = ColumnSelector(
                column_idx = feature_mask.nonzero()[0],
                column_names = self.feature_names_out_,
            )
            X = column_selector.transform(X)
            transforms.append(('dropcolumns', column_selector))
        else:
            feature_mask = np.ones(X.shape[1], dtype = 'bool')

        model = LinearRegression(
            fit_intercept = self.fit_intercept,
        )
        model.fit(X, y, sample_weight = sample_weights)
        if transforms:
            self.model = Pipeline([*transforms, ('model', model)])
            self._is_pipeline = True
        else:
            self.model = model
            self._is_pipeline = False

        self._coef[feature_mask] = model.coef_

        self._is_fit = True
        return self


    def __iter__(
        self,
    ) -> Iterator:

        if not self._is_fit:
            raise TypeError('fit must be called before __iter__.')
        elif self._is_pipeline:
            return iter(self.model)
        else:
            return iter((self.model,))


    def __getitem__(
        self,
        *args,
    ):
        if not self._is_fit:
            raise TypeError('fit must be called before __getitem__.')
        elif not self._is_pipeline:
            return (self.model,).__getitem__(*args)
        else:
            return self.model.__getitem__(*args)


    def predict(
        self,
        X: ArrayType2D,
        transform: bool = True,
    ) -> NDArray[np.number]:

        if not transform:
            return self[-1].predict(X)
        else:
            return self.model.predict(X)


    def transform(
        self,
        X: ArrayType2D,
    ) -> NDArray[np.number] | pd.DataFrame:

        if not self._is_fit:
            raise TypeError('fit must be called before transform.')

        X, has_columns = _handle_X_shape_type(X)
        if not self._is_pipeline:
            return X
        else:
            for transformer in self.model[:-1]:
                X = transformer.transform(X)
            return X


    def score(
        self,
        X: ArrayType2D,
        y: ArrayType1D,
        metric: string_scorers_literal = 'R2',
        transform: bool = True,
    ) -> float:

        y_pred = self.predict(X, transform = transform)
        return next(iter(score_from_string(
            y,
            y_pred,
            metric,
        ).values()))


    @property
    def coef_(
        self
    ) -> NDArray[np.number]:

        if not self._is_fit:
            raise TypeError('fit must be called before coef_.')
        return self._coef


    @property
    def intercept_(
        self
    ) -> float:

        if not self._is_fit:
            raise TypeError('fit must be called before intercept_.')
        return self[-1].intercept_


    @property
    def coef_intercept_(
        self,
    ) -> NDArray[np.number]:

        if not self._is_fit:
            raise TypeError('fit must be called before coef_.')
        return np.append(self._coef, self[-1].intercept)


    def get_feature_names_out_(
        self,
    ) -> ArrayType1D | None:

        return self.feature_names_out_



@beartype
def fit_linear_model(
    X: pd.DataFrame,
    y: pd.Series,
    p_threshold: Number,
    polynomial_degree: int = 1,
    fit_intercept: bool = True,
    sample_weights: NDArray[np.number] | None = None,
    test_size = 0.2,
    scorers: Iterable[string_scorers_literal] = ['R2', 'MAE'],
    random_state: int | None = None,
    stratify: NDArray | None = None,
    groups: NDArray | None = None,
    train_idx: NDArray[np.integer] | list[int] | None = None,
) -> tuple[LinearModel, dict[str, Number], NDArray[np.integer]]:

    if sample_weights is None:
        sample_weights = np.ones(y.shape[0])

    if train_idx is None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = general_split(
            X, 
            y, 
            sample_weights,
            split_ratio = [1-test_size, test_size],
            random_state = random_state,
            groups = groups,
            stratify = stratify,
        )
    else:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        test_idx = X.index.drop(X_train.index)
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        if sample_weights is not None:
            weights_train = sample_weights[train_idx]
            weights_test = np.delete(sample_weights, train_idx)

    model = LinearModel(
        polynomial_degree = polynomial_degree,
        fit_intercept = fit_intercept,
        p_threshold = p_threshold,
    )
    model.fit(X_train, y_train, sample_weights = weights_train)
    y_train_pred = model.predict(X_train, transform = True)
    y_test_pred = model.predict(X_test, transform = True)

    scores = {f'training {key}': val for key, val in score_from_string(
        y_train, 
        y_train_pred, 
        scorers,
    ).items()} | {f'test {key}': val for key, val in score_from_string(
        y_test, 
        y_test_pred,
        scorers,
    ).items()}

    return (model, scores, (X.index.isin(X_train.index.values)).nonzero()[0])