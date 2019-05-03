import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as ff

from sklearn.metrics import roc_curve, auc

import numpy as np
import pandas as pd


def plot_histogram(data: list, names: list, lines: list = None, class_name: str = None, hist: bool = False) -> dcc.Graph:
    if not hist:
        shapes = None
        if lines is not None:
            print(lines)
            shapes = []
            for i in range(len(lines) - 1):
                shapes.append({
                    'type': 'rect',
                    'xref': 'x',
                    'yref': 'paper',
                    'x0': lines[i],
                    'y0': 0,
                    'x1': lines[i + 1],
                    'y1': 1,
                    'fillcolor': '#d3d3d3' if i % 2 == 0 else '#ffffff',
                    'opacity': 0.2,
                    'line': {
                        'width': 0,
                    }
                })

        fig = ff.create_distplot(data, names, show_hist=False)
        fig.layout.shapes = shapes
    else:
        traces = []
        for i in range(len(data)):
            traces.append(go.Bar(
                x=lines,
                y=[np.sum(np.where(data[i] == v, 1, 0)) for v in lines],
                name=names[i]
            ))
        layout = go.Layout(barmode='stack', xaxis=dict(tickmode='array', tickvals=lines))
        fig = go.Figure(data=traces, layout=layout)

    return dcc.Graph(
        figure=fig,
        className=class_name
    )


def plot_scatter(y, x: np.array, names: list = None, mode: str = 'markers', multiple: bool = False,
                 title: str = '', xtitle: str = '', ytitle: str = '') -> dcc.Graph:
    data = []
    if multiple:
        for i, serie in enumerate(y):
            data.append(
                go.Scatter(
                    x=x,
                    y=serie,
                    mode=mode,
                    name=names[i]
                )
            )
    else:
        data = go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=names
        )
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xtitle),
        yaxis=dict(title=ytitle)
    )
    return dcc.Graph(figure=go.Figure(data=data, layout=layout))


def plot_boxplot(scores: pd.DataFrame, col: str, title: str = '', xlabel : str = '', ylabel : str = '') -> dcc.Graph:
    traces = []
    for v in scores[col].unique():
        traces.append(
            go.Box(y=scores.loc[scores[col] == v, 'z1'],
                   name=str(v),
                   boxpoints='all',
                   marker={'size': 2},
                   line={'width': 1}))
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel)
    )
    return dcc.Graph(figure=go.Figure(data=traces, layout=layout))


def plot_roc_curve(y_true: pd.Series, thetas: pd.Series, pos_value, line_width: int = 2, legend_text: str = '',
                   xlabel: str = '', ylabel: str = '', ylabel2:  str = '', title:  str = '') -> dcc.Graph:
    fpr, tpr, thresholds = roc_curve(y_true, thetas, pos_label=pos_value)
    roc_auc = auc(fpr, tpr)

    trace1 = go.Scatter(x=fpr, y=tpr,
                        mode='lines',
                        line=dict(color='darkorange', width=line_width),
                        name=legend_text + ' (area = {0:.2f})'.format(roc_auc)
                        )

    trace2 = go.Scatter(x=[0, 1], y=[min(thresholds), max(thresholds)],
                        mode='lines',
                        line=dict(color='navy', width=line_width, dash='dash'),
                        showlegend=False,
                        yaxis='y2')

    layout = go.Layout(title=title,
                       xaxis=dict(title=xlabel),
                       yaxis=dict(title=ylabel),
                       yaxis2=dict(title=ylabel2, overlaying='y', side='right', zeroline=False,
                                   showgrid=False))

    return dcc.Graph(figure=go.Figure(data=[trace1, trace2], layout=layout))


def group_model_plots(icc_plot_list: list, boxplots: list, iic_plot: dcc.Graph):
    icc_side_by_side = []
    for i in range(0, int(np.ceil(len(icc_plot_list)/2))):
        if (1 + (i * 2)) < len(icc_plot_list):
            icc_side_by_side.append(html.Div([
                icc_plot_list[i*2],
                icc_plot_list[1 + (i * 2)]
            ], className="six columns"))
        else:
            icc_side_by_side.append(html.Div([
                icc_plot_list[i * 2]
            ], className="six columns"))

    boxplots_side_by_side = []
    for i in range(0, int(np.ceil(len(boxplots)/2))):
        if (1 + (i * 2)) < len(boxplots):
            boxplots_side_by_side.append(html.Div([
                boxplots[i*2],
                boxplots[1 + (i * 2)]
            ], className="six columns"))
        else:
            boxplots_side_by_side.append(html.Div([
                boxplots[i * 2]
            ], className="six columns"))

    return [
        html.Div(icc_side_by_side, className='row'),
        html.Div(boxplots_side_by_side, className='row'),
        html.Div(iic_plot, className='twelve columns')
    ]
