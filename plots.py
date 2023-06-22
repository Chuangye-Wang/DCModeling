import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 12


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


# def plot_DvsX(diffusivity_data, literature_list, diffusion_type, ax, axL, xlim2 = [], **kwargs):
#     '''
#     Dtype : string
#         Type of diffusion coefficients to plot. 'DTA', 'DTB', 'DIA', 'DIB', 'DC'
#     '''
#     # mpl.rcParams['lines.linestyle'] = '--'
#     df = diffusivity_data.data.copy()
#     df = df[df['Literature'].isin(literature_list)]
#     d_type, d_elem = diffusion_type[:2], diffusion_type[2:]
#     df = df[df['Dtype'] == d_type]
#     if len(Dtype) > 2:
#         exp = exp[exp['Element'] == Dtype[2]]
#
#     # print(exp)
#     temperature_list = exp['T_C'].unique()
#     if not exp.empty:
#         fit = df_model[df_model['T_C'].isin(temperature_list)].copy()
#     else:
#         fit = df_model
#
#     exp.rename(columns={'T_C': 'T $\degree$C'}, inplace=True)
#     fit.rename(columns={'T_C': 'T $\degree$C'}, inplace=True)
#     exp.rename(columns={'B_mp': Elements[1] + ' at.%'}, inplace=True)
#     fit.rename(columns={'B_mp': Elements[1] + ' at.%'}, inplace=True)
#
#     if B_xlim:
#         if len(B_xlim) != 2:
#             raise ValueError("Length of B_xlim is not correct, should be 2.")
#         xlim_low, xlim_high = B_xlim
#         fit = fit[(fit[Elements[1] + ' at.%'] >= xlim_low) & (fit[Elements[1] + ' at.%'] <= xlim_high)]
#
#     if not exp.empty:
#         sns.lineplot(data=fit, x=Elements[1] + ' at.%', y=Dtype + '_' + Model, hue='T $\degree$C', legend=False, ax=ax,
#                      **kwargs)
#         sns.scatterplot(data=exp, x=Elements[1] + ' at.%', y='Dexp', hue='T $\degree$C', style='Literature', s=s, ax=ax,
#                         legend="full", **kwargs)
#     else:
#         # assume fit is not empty.
#         sns.lineplot(data=fit, x=Elements[1] + ' at.%', y=Dtype + '_' + Model, hue='T $\degree$C', legend="full", ax=ax,
#                      **kwargs)
#     ax.set_yscale('log')
#     ax.set_ylabel('D (m$^2$/s)')
#     if B_xlim:
#         ax.set_xlim(B_xlim[0], B_xlim[1])
#     else:
#         ax.set_xlim(0, 100)
#     # ax.set_ylim(low, up)
#
#     handles, labels = ax.get_legend_handles_labels()
#     axL.legend(handles=handles, labels=labels, loc='best', framealpha=1, edgecolor='w', ncol=int(len(handles) / 10)+1)
#     axL.axis('off')
#     ax.get_legend().remove()
#     mpl.rcParams['lines.linestyle'] = '-'
