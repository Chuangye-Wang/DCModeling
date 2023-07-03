import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import constants

matplotlib.rcdefaults()
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['ytick.left'] = True
matplotlib.rcParams['xtick.major.size'] = 6
matplotlib.rcParams['ytick.major.size'] = 6
matplotlib.rcParams['lines.markersize'] = 10
matplotlib.rcParams['xtick.minor.size'] = 2
matplotlib.rcParams['ytick.minor.size'] = 2
matplotlib.rcParams['xtick.labelsize'] = 12


# matplotlib.rcParams['ytick.labelsize'] = 12
# matplotlib.rcParams['axes.labelsize'] = 12


def exp_vs_pred_plot(df, model, ax=None, ax_legend=None, off_value=3, logx=True, logy=True, show_mse=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if ax_legend is None:
        fig_legend, ax_legend = plt.subplots()
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = "black"
    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 0.5
    if logx:
        ax.set_yscale('log')
    if logy:
        ax.set_xscale('log')

    low = 10 ** np.floor(np.log10(min(df[df['D_' + model] > 0][['Dexp', 'D_' + model]].min())))
    up = 10 ** np.ceil(np.log10(max(df[['Dexp', 'D_' + model]].max())))
    ax.plot([low, up], [low, up], c='k')
    ax.plot([low / off_value, up / off_value], [low, up], 'k:')
    ax.plot([low * off_value, up * off_value], [low, up], 'k:')
    ax.set_xlim(low, up)
    ax.set_ylim(low, up)
    ax.set_xlabel('Experimental D (m$^2$/s)')
    ax.set_ylabel('Predicted D (m$^2$/s)')

    sns.scatterplot(data=df, x='Dexp', y='D_' + model, ax=ax, **kwargs)

    if show_mse:
        mse = np.sum(np.log(df['D_' + model] / df['Dexp']) ** 2) / df['Dexp'].size
        ax.text(0.75, 0.15, 'MSE:', transform=ax.transAxes, fontsize=10)
        ax.text(0.82, 0.15, '{:.3f}'.format(mse), transform=ax.transAxes, fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    if ax_legend is not None:
        ax_legend.legend(handles=handles, labels=labels, loc='best', framealpha=1, edgecolor='w')
        ax_legend.axis('off')

    ax.get_legend().remove()


def func():
    return False


def conditions_plot(diffusion_data, grid_data, literature_list, x_type, ax, ax_legend, diffusion_type="DT",
                    element="A", x_axis_element="B", inv_temp_numerator=1E4, **kwargs):
    '''
    Dtype : string
        Type of diffusion coefficients to plot. 'DTA', 'DTB', 'DIA', 'DIB', 'DC'
    '''
    df_exp = diffusion_data.data.copy()
    elements = diffusion_data.elements
    df_grid = grid_data.copy()
    df_exp = df_exp[df_exp['Literature'].isin(literature_list) & (df_exp['Dtype'] == diffusion_type)]
    if element in {"A", "B"}:
        df_exp = df_exp[df_exp["Element"] == element]

    if x_type.lower() == "temperature":
        temperature_list = df_exp['temp_celsius'].unique()
        df_grid = df_grid[df_grid['temp_celsius'].isin(temperature_list)]

        if not df_exp.empty:
            sns.scatterplot(data=df_exp, x=f"comp_{x_axis_element}_mf", y='Dexp', hue='temp_celsius',
                            style='Literature',
                            ax=ax, legend="full", **kwargs)
            sns.lineplot(data=df_grid, x=f"comp_{x_axis_element}_mf", y=diffusion_type + element, hue='temp_celsius',
                         ax=ax,
                         legend=False, linestyle="--", **kwargs)
        else:
            # assume fit is not empty.
            sns.lineplot(data=df_grid, x=f"comp_{x_axis_element}_mf", y=diffusion_type + element, hue='temp_celsius',
                         ax=ax, legend="full", linestyle="--", **kwargs)
        ax.set_xlabel(f'Mole fraction of {elements[constants.ELEMENTS_ORDER[x_axis_element]]}')

    elif x_type.lower() == "composition":
        composition_list = df_exp['comp_A_mf'].unique()
        df_grid = df_grid[df_grid['comp_A_mf'].isin(composition_list)]
        df_exp["reverse_T"] = df_exp["temp_kelvin"].mul(1 / inv_temp_numerator)
        df_grid["reverse_T"] = df_grid["temp_kelvin"].mul(1 / inv_temp_numerator)

        if not df_exp.empty:
            sns.scatterplot(data=df_exp, x="reverse_T", y='Dexp', hue='comp_A_mf',
                            style='Literature',
                            ax=ax, legend="full", **kwargs)
            sns.lineplot(data=df_grid, x="reverse_T", y=diffusion_type + element, hue='comp_A_mf',
                         ax=ax, legend=False, linestyle="--", **kwargs)
        else:
            # assume fit is not empty.
            sns.lineplot(data=df_grid, x="reverse_T", y=diffusion_type + element, hue='comp_A_mf',
                         ax=ax, legend="full", linestyle="--", **kwargs)
        ax.set_xlabel(f'{inv_temp_numerator}/T (K$^{-1}$)')

    ax.set_ylabel(f'{elements[constants.ELEMENTS_ORDER[element]]} '
                  f'{constants.DIFFUSION_TYPES[diffusion_type]} D (m$^2$/s)')
    low = 10 ** np.floor(np.log10(min(df_exp["Dexp"].min(),
                                      df_grid[df_grid[diffusion_type + element] > 0][diffusion_type + element].min())))
    up = 10 ** np.ceil(np.log10(max(df_exp["Dexp"].max(), df_grid[diffusion_type + element].max())))
    ax.set_ylim(low, up)
    ax.set_yscale('log')

    # ax_legend.get_legend().get_texts()[0].set_text("T $\degree$C")
    handles, labels = ax.get_legend_handles_labels()
    if x_type.lower() == "temperature":
        labels[0] = "T $\degree$C"
    elif x_type.lower() == "composition":
        labels[0] = f"{elements[constants.ELEMENTS_ORDER[element]]} mol/mol"
    ax_legend.legend(handles=handles, labels=labels, loc='best', framealpha=1, edgecolor='w',
                     ncol=int(len(handles) / 15) + 1)
    ax_legend.axis('off')
    ax.get_legend().remove()
