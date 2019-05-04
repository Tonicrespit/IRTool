import base64
import io
import os
import math
import numpy as np

from Views import visibility_manager
from Views import div_view
from Views import table_view
from Views import tab_view
from Views import graph_view

import App

from Models.datasource import DataSource


class DataSourceController:
    def __init__(self, app: App):
        self.data = None
        self.tab_column_mapping = {}
        self.app = app
        self.log_callbacks()

    def log_callbacks(self):
        self.app.set_callbacks([
            self.app.define_callback(
                input=[('data-file-upload', 'contents')],
                state=[('data-file-upload', 'filename')],
                output=('data-test-train-split-container', 'style'),
                func=self.on_file_select
            ),
            self.app.define_callback(
                input=[('data-test-train-split-container', 'style')],
                output=('data-test-train-split-select-dropdown', 'options'),
                func=self.fill_possible_targets
            ),
            self.app.define_callback(
                input=[('data-test-train-split-submit', 'n_clicks')],
                state=[('data-test-train-split-select-dropdown', 'value'),
                       ('data-test-train-split-train-size', 'value'),
                       ('data-test-train-split-random-state', 'value'),
                       ('data-test-train-split-stratify', 'values')],
                output=('data-summary-container', 'style'),
                func=self.on_train_test_split
            ),
            self.app.define_callback(
                input=[('data-summary-container', 'style')],
                output=('data-summary-tabs', 'value'),
                func=self.on_data_load
            ),
            self.app.define_callback(
                input=[('data-summary-container', 'style')],
                output=('columns-to-use-checklist', 'options'),
                func=self.fill_columns_checklist
            ),
            self.app.define_callback(
                input=[('data-summary-tabs', 'value')],
                output=('data-summary-tab-content', 'children'),
                func=self.on_tab_change
            ),
            self.app.define_callback(
                input=[('columns-to-use-checklist', 'options')],
                output=('columns-to-use-checklist', 'values'),
                func=self.default_column_selection
            ),
            self.app.define_callback(
                input=[('columns-to-use-checklist', 'values')],
                output=('target-value-dropdown', 'options'),
                func=self.update_column_selection
            ),
            self.app.define_callback(
                input=[('target-value-dropdown', 'value')],
                output=('data-categorize-parent', 'style'),
                func=self.on_target_value_select
            ),
            self.app.define_callback(
                input=[('data-categorize-parent', 'style')],
                output=('data-categorize-tabs', 'children'),
                func=self.show_data_categorization_tabs
            ),
            self.app.define_callback(
                input=[('data-categorize-tabs', 'children')],
                output=('data-categorize-tabs', 'value'),
                func=self.on_data_categorization_show
            ),
            self.app.define_callback(
                input=[('data-categorize-tabs', 'value')],
                output=('categorized-column-preview', 'children'),
                func=self.on_caategorize_tab_select
            ),
            self.app.define_callback(
                input=[('categorize-data-button', 'n_clicks')],
                output=('model-output-container', 'style'),
                func=self.on_categorize_button_click
            ),
        ])

    def on_file_select(self, data: str, filename: str):
        if data is None:
            return visibility_manager.hide()

        name, extension = os.path.splitext(filename)
        content_type, content_string = data.split(',')

        decoded = base64.b64decode(content_string)
        try:
            if extension in ['.csv']:
                # Assume that the user uploaded a CSV file
                self.data = DataSource(io.StringIO(decoded.decode('utf-8')), extension[1:])
            elif extension in ['.xls', '.xlsx']:
                # Assume that the user uploaded an excel file
                self.data = DataSource(io.BytesIO(decoded), extension[1:])
            else:
                return False
        except Exception as e:
            print(e)
            return visibility_manager.hide()
        return visibility_manager.show()

    def fill_possible_targets(self, state):
        if state is None or visibility_manager.is_hidden(state):
            return []
        return [{'label': i, 'value': i} for i in self.data.get_columns() if self.data.is_column_categorical(i)]

    def on_train_test_split(self, n_clicks: int, target: list, train_size: str, rnd_state: str, stratify: list) -> dict:
        if n_clicks is None or train_size is None or n_clicks == 0:
            return visibility_manager.hide()

        if len(stratify) > 0 and target is None:
            # To stratify, a target is needed
            return visibility_manager.hide()

        stratify = stratify is not None and len(stratify) > 0
        shuffle = True
        try:
            if train_size is not None:
                train_size = float(train_size)
            else:
                train_size = 1
            if rnd_state is not None:
                rnd_state = int(rnd_state)
            else:
                rnd_state = None
        except ValueError:
            return visibility_manager.hide()

        self.data.set_target(target)
        self.data.test_train_split(train_size, rnd_state, stratify, shuffle)
        return visibility_manager.show()

    def on_data_load(self, state: dict):
        if state is None or visibility_manager.is_hidden(state):
            return 'no-tab'
        return 'tabs-1-preview'

    def on_tab_change(self, tab: str):
        if tab is None:
            return []

        elif tab == 'tabs-1-summary':
            summary = self.data.get_data()[[c for c in self.data.get_columns() if not self.data.is_column_categorical(c)]].describe()
            summary['metric'] = summary.index
            column_order = ['metric'] + list(self.data.get_data()[[c for c in self.data.get_columns() if not self.data.is_column_categorical(c)]].describe().columns)

            return table_view.create_table(summary, column_order)
        elif tab == 'tabs-1-preview':
            data = self.data.get_data_as_str()
            return table_view.create_table(data)

    def fill_columns_checklist(self, state):
        if state is None or visibility_manager.is_hidden(state):
            return []

        return [] if state is None else [{'label': c, 'value': c} for c in self.data.get_columns()]

    def default_column_selection(self, cols):
        if cols is None or len(cols) == 0:
            return []

        return [c['value'] for c in cols]

    def update_column_selection(self, cols):
        if cols is None or len(cols) == 0:
            return []

        self.data.set_columns(cols)
        possible_values = self.data.get_target().unique()
        return [{'label': i, 'value': i} for i in possible_values]

    def on_target_value_select(self, target_value):
        if target_value is None:
            return visibility_manager.hide()

        self.data.set_target_value(target_value)
        return visibility_manager.show()

    def show_data_categorization_tabs(self, state):
        if visibility_manager.is_hidden(state):
            return []
        tabs = []
        for column in self.data.get_columns():
            if not self.data.get_target_column() == column:
                self.tab_column_mapping['tabs-2-' + column] = column
                tabs.append(tab_view.create_tab(column, value='tabs-2-'+column))
        return tabs

    def on_data_categorization_show(self, state):
        if state is None or len(state) <= 0:
            return 'no-tab'
        return state[0]['props']['value']

    def on_caategorize_tab_select(self, tab):
        if tab == 'no-tab' or len(tab) <= 0:
            return []
        col = self.tab_column_mapping[tab]

        h = graph_view.plot_histogram(self.data.get_column_by_target_value(col),
                                      [str(self.data.get_target_value()),
                                       self.app.lang.translate('plots.not') + str(self.data.get_target_value())])
        target_x = h.figure.data[0].x
        target_y = h.figure.data[0].y
        other_x = h.figure.data[1].x
        other_y = h.figure.data[1].y
        self.data.default_column_categorization(col, target_x, target_y, other_x, other_y)

        category_summary = self.data.preview_categorization(col)
        precision, recall = self.data.compute_cuts_precision_recall(col)
        if len(self.data.get_target().unique()) > 2:
            labels = [str(self.data.get_target_value()),
                      self.app.lang.translate('plots.not') + str(self.data.get_target_value())]
        else:
            labels = self.data.get_target().unique().tolist()
        return [
            table_view.create_table(category_summary),
            div_view.div([
                graph_view.plot_histogram(self.data.get_column_by_target_value(col),
                                          labels,
                                          self.data.get_column_categories(col),
                                          hist=self.data.is_column_categorical(col),
                                          class_name='nine columns'),
                div_view.div([
                    div_view.div(self.app.lang.translate('plots.precision') + ': {0:.4f}'.format(precision)),
                    div_view.div(self.app.lang.translate('plots.recall') + ': {0:.4f}'.format(recall))
                ],
                class_name='three columns')
            ], class_name='row')

        ]

    def on_categorize_button_click(self, n_clicks):
        if n_clicks is None or n_clicks <= 0:
            return visibility_manager.hide()
        self.data.categorize()
        return visibility_manager.show()
