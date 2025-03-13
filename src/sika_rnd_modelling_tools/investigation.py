from .utilities import get_group_means_stds

import operator as op
from numpy.typing import NDArray

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

import xgboost as xgb

import shap
from xgbfir.main import XgbModelParser


def get_importances_by_group(
    xgb_model: xgb.Booster,
    importance_data: pd.DataFrame,
    groups: NDArray,
    remove_no_importance_features: bool = True,
) -> tuple[dict, int]:
    

    explainer = shap.TreeExplainer(
        xgb_model,
        importance_data,
    )
    importances = np.abs(explainer(importance_data).values)
    features_out = importance_data.columns.values

    if remove_no_importance_features:
        importances_mask = (importances > 0).any(axis = 0)
        importances = importances[:,importances_mask]
        features_out = features_out[importances_mask]

    return get_group_means_stds(importances, groups), features_out


def get_feature_interactions_xgb(
    xgb_model: xgb.Booster,
    depth: int = 1,
) -> pl.DataFrame:

    interactions = XgbModelParser().GetXgbModelFromMemory(
        xgb_model.get_dump('', with_stats = True), 
        3,
    ).GetFeatureInteractions(
        depth, 
        -1,
    ).GetFeatureInteractionsOfDepth(
        depth,
    )
    return pl.LazyFrame(
        map(
            op.attrgetter(
                'Name',
                'Gain',
                'FScore',
                'FScoreWeighted',
                'AverageFScoreWeighted',
                'AverageGain',
                'ExpectedGain',
                'AverageTreeIndex',
                'AverageTreeDepth',
            ),
            interactions,
        ),
        schema = {
            'name': pl.String,
            'gain': pl.Float32,
            'fscore': pl.Float32,
            'weighted fscore': pl.Float32,
            'mean weighted fscore': pl.Float32,
            'mean gain': pl.Float32,
            'expected gain': pl.Float32,
            'mean tree index': pl.Float32,
            'mean depth': pl.Float32,
        }
    ).select(
        pl.col(
            'name'
        ).str.split_exact(
            '|',
            1,
        ).struct.rename_fields(
            [
                'feature 1',
                'feature 2',
            ]
        ).struct.unnest(),
        cs.exclude(
            'name'
        ),
    ).with_columns(
        pl.sum_horizontal(
            pl.col(
                'weighted fscore',
                'gain',
            ).arg_sort().add(
                60
            ).cast(
                pl.Float32,
            ).pow(
                -1
            )
        ).arg_sort().alias(
            'rank'
        )
    ).sort(
        by = 'rank'
    ).collect()