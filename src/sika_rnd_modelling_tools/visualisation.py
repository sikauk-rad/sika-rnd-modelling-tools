from matplotlib import pyplot as plt
from numpy.typing import NDArray
from typing import Literal
from collections.abc import Iterable
from beartype import beartype
import numpy as np
from sklearn import metrics
from .estimators import LinearModel
from sklearn.linear_model import LinearRegression
import seaborn as sns
from .types import ArrayType1D
from numbers import Number
from colorsys import rgb_to_hls, hls_to_rgb
from matplotlib.colors import ColorConverter
from .utilities import string_scorers_literal, score_from_string

SIKA_YELLOW = '#FFC510'
SIKA_RED = '#E61F26'
SIKA_GREEN = '#468283'
SIKA_BLUE = '#50758D'


@beartype
def scale_lightness(
    hex_colour: str, 
    scale_l: float,
) -> tuple[int, int, int]:

    h, l, s = rgb_to_hls(*ColorConverter.to_rgb(hex_colour))
    return hls_to_rgb(h, min(1, l * scale_l), s = s)


@beartype
def plot_predictions(
    y_train: NDArray[np.number],
    y_train_pred: NDArray[np.number],
    y_test: NDArray[np.number],
    y_test_pred: NDArray[np.number],
    ax: plt.Axes,
    y_valid: NDArray[np.number] | None = None,
    y_valid_pred: NDArray[np.number] | None = None,
):

    train_r2 = metrics.r2_score(y_train, y_train_pred)
    test_r2 = metrics.r2_score(y_test, y_test_pred)
    ax.scatter(
        y_train, 
        y_train_pred, 
        marker = 'o', 
        edgecolor = 'black',
        c = 'blue', 
        linewidth = 0.5,
        s = 40, 
        label = f'training data (R² = {train_r2:.2f})',
    )
    if y_valid is not None and y_valid_pred is not None:
        valid_r2 = metrics.r2_score(y_valid, y_valid_pred)
        ax.scatter(
            y_valid,
            y_valid_pred,
            marker = '^',
            linewidth = 0.5,
            c = SIKA_YELLOW,
            s = 40,
            label = f'validation data (R² = {valid_r2:.2f})',
        )
    ax.scatter(
        y_test, 
        y_test_pred, 
        marker = 'x', 
        linewidth = 2,
        c = SIKA_RED, 
        s = 40,
        zorder = 10,
        label = f'test data (R² = {test_r2:.2f})',
    )
    axmin = min(y_train.min(), y_train_pred.min(), y_test.min(), y_test_pred.min())
    axmax = max(y_train.max(), y_train_pred.max(), y_test.max(), y_test_pred.max())
    if y_valid is not None and y_valid_pred is not None:
        axmin = min(y_valid.min(), y_valid_pred.min(), axmin)
        axmax = max(y_valid.max(), y_valid_pred.max(), axmax)

    axmin *= 0.95
    axmax *= 1.05

    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.plot(
        [axmax, axmin][::-1],
        [axmin, axmax],
        marker = 'none',
        linestyle = 'dashed',
        color = 'black',
        label = None,
    )
    ax.legend()


@beartype
def plot_linear_model(
    linear_model: LinearModel | LinearRegression,
    y_train: ArrayType1D,
    y_train_pred: ArrayType1D,
    y_test: ArrayType1D,
    y_test_pred: ArrayType1D,
    y_name: str,
    figsize: tuple[int, int] = (15,7),
    coef_plot_limit: Number | None = 0,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:

    fig, (ax_scatter, ax_bar) = plt.subplots(
        ncols = 2, 
        figsize = figsize,
        sharex = False,
        sharey = False,
    )
    if coef_plot_limit is not None:
        coef_mask = np.abs(linear_model.coef_) > coef_plot_limit
    else:
        coef_mask = np.ones(linear_model.coef_.shape[0], dtype = 'bool')

    ax_bar.bar(
        linear_model._feature_names_in[coef_mask],
        linear_model.coef_[coef_mask],
        edgecolor = 'black',
        color = np.where(linear_model.coef_[coef_mask] > 0, SIKA_YELLOW, SIKA_RED),
    )
    ax_bar.set_title('coefficients')
    ax_bar.tick_params(axis = 'x', labelrotation = 90)

    plot_predictions(
        y_train, 
        y_train_pred, 
        y_test,
        y_test_pred,
        ax_scatter,
    )

    ax_scatter.set_xlabel(f'actual {y_name}')
    ax_scatter.set_ylabel(f'predicted {y_name}')
    ax_scatter.legend()
    plt.show()

    return fig, (ax_scatter, ax_bar)


@beartype
def generate_score_string(
    y: ArrayType1D,
    y_pred: ArrayType1D,
    scorers: string_scorers_literal | list[string_scorers_literal],
) -> str:

    return ', '.join(
        [f'{metric} = {score:.3g}' for metric, score in score_from_string(y, y_pred, scorers).items()]
    )


@beartype
def plot_regression_results(
    y_train: ArrayType1D, 
    y_test: ArrayType1D, 
    y_train_pred: ArrayType1D, 
    y_test_pred: ArrayType1D,
    y_name: str,
    y_valid: ArrayType1D | None = None,
    y_valid_pred: ArrayType1D | None = None,
    train_colour: str = SIKA_BLUE,
    valid_colour: str = SIKA_GREEN,
    test_colour: str = SIKA_YELLOW,
    figure_size: tuple[int, int] = (10, 7),
    bins: int | None | ArrayType1D = 10,
    include_lines: bool = True,
    include_xy: bool = True,
    style: Literal['darkgrid', 'whitegrid', 'ticks', 'dark', 'white'] = 'darkgrid',
    context: Literal['paper', 'notebook', 'talk', 'poster'] = 'poster',
    scorers: string_scorers_literal | Iterable[string_scorers_literal] = ['R2'],
) -> sns.JointGrid:

    has_valid = not not (y_valid is not None) and (y_valid_pred is not None)
    hist_data_true = np.concatenate([y_train, y_valid, y_test] if has_valid else [y_train, y_test])
    hist_data_pred = np.concatenate([y_train_pred, y_valid_pred, y_test_pred] if has_valid else [y_train_pred, y_test_pred])
    all_hist_data = np.concatenate((hist_data_true, hist_data_pred))        

    with sns.axes_style(style), sns.plotting_context(context):
        grid_plot = sns.JointGrid(
            height = 8,
        )
        sns.regplot(
            x = y_train,
            y = y_train_pred,
            ax = grid_plot.ax_joint,
            label = f'training data ({generate_score_string(y_train, y_train_pred, scorers)})',
            marker = 'o',
            scatter_kws = dict(
                edgecolor = 'white',
                linewidths = 1,
                s = 100,
            ),
            line_kws = dict(
                linewidth = 3,
            ),
            fit_reg = include_lines,
            color = train_colour,
        )
        if has_valid:
            sns.regplot(
                x = y_valid,
                y = y_valid_pred,
                ax = grid_plot.ax_joint,
                label = f'validation data ({generate_score_string(y_valid, y_valid_pred, scorers)})',
                marker = 'o',
                scatter_kws = dict(
                    edgecolor = 'white',
                    linewidths = 1,
                    s = 100,
                ),
                line_kws = dict(
                    linewidth = 3,
                ),
                fit_reg = include_lines,
                color = valid_colour,
            )
        sns.regplot(
            x = y_test,
            y = y_test_pred,
            ax = grid_plot.ax_joint,
            label = f'test data ({generate_score_string(y_test, y_test_pred, scorers)})',
            marker = 'o',
            scatter_kws = dict(
                edgecolor = 'white',
                linewidths = 1,
                s = 100,
                # c = SIKA_GREEN,
            ),
            line_kws = dict(
                linewidth = 3,
            ),
            fit_reg = include_lines,
            color = test_colour,
        )

        hues = np.full(hist_data_true.shape[0], 2, dtype = 'int8')
        hues[:y_train.shape[0]] = 0
        if has_valid:
            hues[y_train.shape[0]:-y_test.shape[0]] = 1
        palette = {
            0: train_colour,
            1: valid_colour,
            2: test_colour,
        }
        sns.histplot(
            x = hist_data_true,
            fill = True,
            linewidth = 1,
            ax = grid_plot.ax_marg_x,
            hue = hues,
            multiple = 'stack',
            kde = True,
            palette = palette,
            legend = False,
            bins = bins,
        )
        sns.histplot(
            y = hist_data_pred,
            fill = True,
            linewidth = 1,
            ax = grid_plot.ax_marg_y,
            hue = hues,
            multiple = 'stack',
            kde = True,
            palette = palette,
            legend = False,
            bins = bins,
        )

        axmin = all_hist_data.min()
        axmax = all_hist_data.max()
        axmax += abs(axmax * 0.05)
        axmin -= abs(axmax * 0.05)
        grid_plot.ax_joint.set_xlim(axmin, axmax)
        grid_plot.ax_joint.set_ylim(axmin, axmax)
        grid_plot.ax_joint.legend()
        grid_plot.ax_joint.set_xlabel(f'actual {y_name}')
        grid_plot.ax_joint.set_ylabel(f'predicted {y_name}')
        if include_xy:
            grid_plot.ax_joint.plot(
                [axmax, axmin][::-1],
                [axmin, axmax],
                marker = 'none',
                linestyle = 'dashed',
                color = 'black',
                label = None,
                alpha = 0.5,
            )
        grid_plot.figure.set_figwidth(figure_size[0])
        grid_plot.figure.set_figheight(figure_size[1])
        return grid_plot


# # PlotEntryType = TypedDict('PlotEntryType', {
# #     'true': ArrayType1D, 
# #     'pred': ArrayType1D, 
# #     'colour': str,
# #     'marker': str,
# # })


# @beartype
# def plot_regression_predictions(
#     # data_true_pred: dict[str, PlotEntryType],

#     ax: plt.axis,
#     scorers: Iterable[string_scorers_literal] = ['R2'],
# ) -> None:


#     for data_label, data_dict in data_true_pred.items():
#         true, pred = itemgetter('true', 'pred')(data_dict)
#         label = ', '.join([f'{data_label} {scorer}: {string_scorers[scorer](true, pred):.2f}' for scorer in scorers])
#         ax.scatter(
#             true,
#             pred,
#             marker = 
#         )


#     ax.scatter(
#         y_train, 
#         y_train_pred, 
#         marker = 'o', 
#         edgecolor = 'black',
#         c = 'blue', 
#         linewidth = 0.5,
#         s = 40, 
#         label = f'training data (R² = {train_r2:.2f})',
#     )
#     if y_valid is not None and y_valid_pred is not None:
#         valid_r2 = metrics.r2_score(y_valid, y_valid_pred)
#         ax.scatter(
#             y_valid,
#             y_valid_pred,
#             marker = '^',
#             linewidth = 0.5,
#             c = SIKA_YELLOW,
#             s = 40,
#             label = f'validation data (R² = {valid_r2:.2f})',
#         )
#     ax.scatter(
#         y_test, 
#         y_test_pred, 
#         marker = 'x', 
#         linewidth = 2,
#         c = SIKA_RED, 
#         s = 40,
#         zorder = 10,
#         label = f'test data (R² = {test_r2:.2f})',
#     )
#     axmin = min(y_train.min(), y_train_pred.min(), y_test.min(), y_test_pred.min())
#     axmax = max(y_train.max(), y_train_pred.max(), y_test.max(), y_test_pred.max())
#     if y_valid is not None and y_valid_pred is not None:
#         axmin = min(y_valid.min(), y_valid_pred.min(), axmin)
#         axmax = max(y_valid.max(), y_valid_pred.max(), axmax)

#     ax.set_xlim(axmin, axmax)
#     ax.set_ylim(axmin, axmax)
#     ax.plot(
#         [axmax, axmin][::-1],
#         [axmin, axmax],
#         marker = 'none',
#         linestyle = 'dashed',
#         color = 'black',
#         label = None,
#     )
#     ax.legend()