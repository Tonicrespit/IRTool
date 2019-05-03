import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd


def create_table(data: pd.DataFrame, columns=None, page_size=5, class_name: str = ''):
    if columns is None:
        columns = data.columns

    return html.Div(
        [
            dash_table.DataTable(
                data=data.to_dict('rows'),
                columns=[{'name': str(i), 'id': i} for i in columns],
                style_table={'overflowX': 'scroll'},
                pagination_settings={
                    'current_page': 0,
                    'page_size': page_size
                }
            )
        ],
        className=class_name
    )
