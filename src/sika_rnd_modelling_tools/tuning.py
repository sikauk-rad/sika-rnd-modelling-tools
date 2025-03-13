import numpy as np
from scipy.stats import uniform, randint, rv_discrete, rv_continuous
from .utilities import import_class
from collections.abc import Callable
from typing import Any, Literal
from numbers import Number
import orjson as json
from importlib.resources import files
from types import NoneType
from beartype import beartype

data_path = files().joinpath('data')


with data_path.joinpath('parameter_grids.json').open(mode = 'rb') as parameter_grids:
    parameter_grids = json.loads(parameter_grids.read())
    model_strings = [*parameter_grids.keys()]

with data_path.joinpath('model_imports.json').open(mode = 'rb') as model_imports:
    model_imports = json.loads(model_imports.read())

with data_path.joinpath('parameter_defaults.json').open(mode = 'rb') as parameter_defaults:
    parameter_defaults = json.loads(parameter_defaults.read())


def get_model_strings() -> list[str]:
    return model_strings


def _check_model_string(
    model_string: str
) -> None:
    if model_string not in parameter_grids.keys():
        raise KeyError(f'model_string {model_string} not found.')


def import_model(
    model_string: str,
):
    model_string = model_string.rstrip(')').rstrip('(')
    return import_class(*model_imports[model_string])


class LogUniformInt(rv_discrete):
    def _pmf(
        self, 
        x: float, 
    ) -> float:
        lower, upper = self.a, self.b
        log_lower, log_upper = np.log(lower), np.log(upper)
        log_x = np.log(x)
        return ((log_lower <= log_x) & (log_x <= log_upper))/x

    
class LogUniformFloat(rv_continuous):
    def _pdf(
        self, 
        x,
    ) -> float:
        return 1.0 / (x * np.log(self.b / self.a))


@beartype
def get_default_parameters(
    model_string: str,
    *,
    n_train_rows: int,
    n_train_cols: int,
    n_classes: int,
    max_boosting_rounds: int,
    early_stopping_rounds: int,
    objective: str | Callable,
    metric: str | Callable,
    monotone_constraints: list[Literal[-1,0,1]] | dict[str, Literal[-1,0,1]] | None = None,
    interaction_constraints: list[list[Number] | list[str]] | None = None,
) -> dict[str, Any]:

    _check_model_string(model_string)
    return {param_name: param_evaluator(
        param,
        n_train_rows = n_train_rows,
        n_train_cols = n_train_cols,
        n_classes = n_classes,
        max_boosting_rounds = max_boosting_rounds,
        early_stopping_rounds = early_stopping_rounds,
        objective = objective,
        metric = metric,
        monotone_constraints = monotone_constraints,
        interaction_constraints = interaction_constraints,
    ) for param_name, param in parameter_defaults[model_string].items()}


@beartype
def param_evaluator(
    param: Any,
    *,
    n_train_rows: int,
    n_train_cols: int,
    n_classes: int,
    max_boosting_rounds: int,
    early_stopping_rounds: int,
    objective: str | Callable,
    metric: str | Callable,
    monotone_constraints: list[Literal[-1,0,1]] | dict[str, Literal[-1,0,1]] | None = None,
    interaction_constraints: list[list[Number] | list[str]] | None = None,
) -> Any:
    if isinstance(param, str):
        if param.startswith('eval '):
            return eval(param[5:])
        elif param in model_strings:
            return import_model(param)
    return param


@beartype
def prepare_parameter_grid(
    model_string: str,
    as_optuna: bool,
    n_train_rows: int,
    n_train_cols: int,
    n_classes: int,
    max_boosting_rounds: int,
    early_stopping_rounds: int,
    objective: str | Callable,
    metric: str | Callable,
    monotone_constraints: list[Literal[-1,0,1]] | dict[str, Literal[-1,0,1]] | None = None,
    interaction_constraints: list[list[Number] | list[str]] | None = None,
) -> tuple[dict[str, Any], int, int]:

    _check_model_string(model_string)
    param_dict = {
        'static': {},
        'categorical': {},
        'int': {},
        'float': {},
    } if as_optuna else {}
    categorical_count = 1
    continuous_count = 0

    for param_name, distribution in parameter_grids[model_string].items():
        if not isinstance(distribution, list):
            distribution = param_evaluator(
                distribution, 
                n_train_rows = n_train_rows,
                n_train_cols = n_train_cols,
                n_classes = n_classes,
                max_boosting_rounds = max_boosting_rounds,
                early_stopping_rounds = early_stopping_rounds,
                objective = objective,
                metric = metric,
                monotone_constraints = monotone_constraints,
                interaction_constraints = interaction_constraints,
            )
            if as_optuna:
                param_dict['static'][param_name] = distribution
            elif isinstance(distribution, (str, Number, NoneType)):
                param_dict[param_name] = [distribution]
            else:
                param_dict[param_name] = distribution
            continue

        distribution = [param_evaluator(
            param, 
            n_train_rows = n_train_rows,
            n_train_cols = n_train_cols,
            n_classes = n_classes,
            max_boosting_rounds = max_boosting_rounds,
            early_stopping_rounds = early_stopping_rounds,
            objective = objective,
            metric = metric,
            monotone_constraints = monotone_constraints,
            interaction_constraints = interaction_constraints,
        ) for param in distribution]

        if (len(distribution) == 3) and isinstance(distribution[2], bool):
            low, high, log = distribution
            if isinstance(low, int) and isinstance(high, int):
                log = log and (low > 0) and (high > 0)
                if as_optuna:
                    param_dict['int'][param_name] = {
                        'name': param_name,
                        'low': low, 
                        'high': high, 
                        'log': log,
                    }
                elif log:
                    param_dict[param_name] = LogUniformInt(low, high)
                else:
                    param_dict[param_name] = randint(low, high + 1)
                continuous_count += 1
                continue

            elif isinstance(low, Number) and isinstance(high, Number):
                log = log and (low > 0) and (high > 0)
                if as_optuna:
                    param_dict['float'][param_name] = {
                        'name': param_name,
                        'low': low, 
                        'high': high, 
                        'log': log,
                    }
                elif log:
                    param_dict[param_name] = LogUniformFloat(low, high)
                else:
                    param_dict[param_name] = uniform(loc = low, scale = high - low)
                continuous_count += 1
                continue

        if as_optuna:
            param_dict['categorical'][param_name] = {
                'name': param_name,
                'choices': distribution,
            }
        else:
            param_dict[param_name] = distribution
        categorical_count *= len(distribution)

    return param_dict, categorical_count, continuous_count