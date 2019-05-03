import Libs.R_wrapper.ltm as ltmlib
import Libs.R_wrapper.utils as utils

from pandas_ml import ConfusionMatrix

from Views import visibility_manager
from Views import tab_view
from Views import table_view
from Views import graph_view
from Views import div_view

import pandas as pd
import numpy as np
import App


class ModelController:
    models = {
        'rasch': ltmlib.rasch,
        'grm': ltmlib.grm,
    }

    def __init__(self, app: App):
        self.app = app
        self.log_callbacks()
        self.last_nclicks = 0
        self.rmodel = None
        self.tab_column_mapping = {}

    def log_callbacks(self):
        self.app.set_callbacks([
            self.app.define_callback(
                input=[('model-output-container', 'style')],
                output=('choose-model-dropdown', 'options'),
                func=self.show_available_models
            ),
            self.app.define_callback(
                input=[('choose-model-dropdown', 'value')],
                output=('test-model-button', 'disabled'),
                func=self.select_model
            ),
            self.app.define_callback(
                input=[('test-model-button', 'n_clicks')],
                state=[('choose-model-dropdown', 'value')],
                output=('model-output', 'style'),
                func=self.run_model
            ),
            self.app.define_callback(
                input=[('model-output', 'style')],
                output=('model-output-roc', 'children'),
                func=self.show_model_roc
            ),
            self.app.define_callback(
                input=[('model-output', 'style')],
                output=('model-output-tabs', 'children'),
                func=self.show_output_tabs
            ),
            self.app.define_callback(
                input=[('model-output-tabs', 'children')],
                output=('model-output-tabs', 'value'),
                func=self.show_default_output_tab
            ),
            self.app.define_callback(
                input=[('model-output-tabs', 'value')],
                state=[('choose-model-dropdown', 'value')],
                output=('model-output-pane', 'children'),
                func=self.show_model_column_graphs
            ),
            self.app.define_callback(
                input=[('choose-threshold-button', 'n_clicks')],
                state=[('model-output-threshold', 'value')],
                output=('test-results', 'children'),
                func=self.show_test_set_results
            ),
        ])

    def init(self, state: list):
        if state is None or len(state) == 0:
            return visibility_manager.hide()

    def show_available_models(self, state: dict):
        return [{'label': model, 'value': model} for model in self.models.keys()]

    def select_model(self, model_value: str):
        if model_value is None:
            return True
        return False

    def run_model(self, n_clicks: int, model_value: str):
        if n_clicks is None or n_clicks <= self.last_nclicks or model_value is None or model_value not in self.models.keys():
            return visibility_manager.hide()
        self.last_nclicks = n_clicks

        self.app.get_data().dropna()
        for col in self.app.get_data().get_without_target().columns:
            self.app.get_data().default_column_itemization(col)

        try:
            self.rmodel = self.models[model_value](self.app.get_data().get_without_target().applymap(str))
        except:
            return visibility_manager.hide()
        return visibility_manager.show()

    def show_model_roc(self, state: dict):
        if state is None or visibility_manager.is_hidden(state):
            return []
        stats = utils.list_to_py(ltmlib.factor_scores(self.rmodel,
                                                      self.app.get_data().get_without_target().applymap(str)))
        stats = stats['score.dat']
        stats['Target'] = self.app.get_data().get_target().tolist()

        return graph_view.plot_roc_curve(
            stats['Target'],
            stats['z1'],
            self.app.get_data().get_target_value(),
            title=self.app.lang.translate('plots.roc_title'),
            xlabel=self.app.lang.translate('plots.roc_xlabel'),
            ylabel=self.app.lang.translate('plots.roc_ylabel'),
            ylabel2=self.app.lang.translate('plots.roc_ylabel2'),
            legend_text=self.app.lang.translate('plots.roc_legend')
        )

    def show_output_tabs(self, state: dict):
        if state is None or visibility_manager.is_hidden(state):
            return []
        prefix = 'tabs-output-'
        tabs = [tab_view.create_tab('general', value=prefix + 'general')]
        self.tab_column_mapping[prefix + 'general'] = 'general'
        for column in self.app.get_data().get_without_target().columns:
            self.tab_column_mapping[prefix + column] = column
            tabs.append(tab_view.create_tab(column, value=prefix + column))
        return tabs

    def show_default_output_tab(self, state):
        if state is None or len(state) <= 0:
            return 'default'
        return list(self.tab_column_mapping.keys())[0]

    def show_model_column_graphs(self, tab: str, model_value: str):
        if tab is None or tab == 'default':
            return []
        col = self.tab_column_mapping[tab]
        if col == 'general':
            scores = utils.list_to_py(ltmlib.factor_scores(self.rmodel,
                                                           self.app.get_data().get_without_target().applymap(str)))['score.dat']
            scores[self.app.get_data().get_target_column()] = self.app.get_data().get_target().tolist()

            rabilities = ltmlib.plot(self.rmodel, model_value, type='IIC', plot=False)

            abilities = utils.to_py(rabilities)
            x = abilities['z']
            columns = [a for a in abilities.columns if a.upper() != 'Z']

            return [
                div_view.div(
                    graph_view.plot_scatter(
                        [abilities[c] for c in columns], x, columns, mode='lines', multiple=True,
                        title=self.app.lang.translate('plots.iic_title'),
                        xtitle=self.app.lang.translate('plots.iic_xlabel'),
                        ytitle=self.app.lang.translate('plots.iic_ylabel')
                    ),
                    class_name='six columns'
                ),
                div_view.div(
                    graph_view.plot_boxplot(
                        scores,
                        self.app.get_data().get_target_column(),
                        title=self.app.lang.translate('plots.boxplot_title') + self.app.get_data().get_target_column(),
                        xlabel=self.app.lang.translate('plots.boxplot_xlabel'),
                        ylabel=self.app.lang.translate('plots.boxplot_ylabel'),
                    ),
                    class_name='six columns'
                )

            ]
        else:
            rcharacteristics = ltmlib.plot(self.rmodel, model_value, plot=False)
            characteristics = utils.list_to_py(rcharacteristics)
            x = characteristics['z']
            cat_responses_df = pd.DataFrame(characteristics['pr'][col])

            scores = utils.list_to_py(ltmlib.factor_scores(self.rmodel, convert=True))['score.dat']

            return [
                div_view.div(
                    graph_view.plot_scatter([cat_responses_df[c] for c in cat_responses_df.columns],
                                            x,
                                            [str(i + 1) for i in range(0, len(cat_responses_df.columns))],
                                            mode='lines',
                                            multiple=True,
                                            title=self.app.lang.translate('plots.icc_title') + col,
                                            xtitle=self.app.lang.translate('plots.icc_xlabel'),
                                            ytitle=self.app.lang.translate('plots.icc_ylabel')),
                    class_name='six columns'
                ),
                div_view.div(
                    graph_view.plot_boxplot(
                        scores,
                        col,
                        title=self.app.lang.translate('plots.boxplot_title') + col,
                        xlabel=self.app.lang.translate('plots.boxplot_xlabel'),
                        ylabel=self.app.lang.translate('plots.boxplot_ylabel'),
                    ),
                    class_name='six columns'
                )
            ]

    def show_test_set_results(self, tab: str, threshold: str):
        if tab is None or tab == 'default' or threshold is None or len(threshold) <= 0:
            return []
        try:
            threshold = float(threshold)
        except ValueError:
            return ['Select a valid threshold.']

        data = self.app.get_data().get_test_set()
        data = data.dropna()
        target = data[self.app.get_data().get_target_column()]
        data = data.drop(self.app.get_data().get_target_column(), axis=1)
        data = data.applymap(str)
        stats = utils.list_to_py(ltmlib.factor_scores(self.rmodel,
                                                      data))
        stats = stats['score.dat']
        stats['Target'] = target.tolist()
        stats['Predicted'] = np.where(stats['z1'] > threshold, 1, 0)

        conf = ConfusionMatrix(stats['Target'], stats['Predicted'])
        print(conf)
        conf_df = conf.to_dataframe()
        conf_df.columns = conf_df.columns.map(int)
        conf_df.index = conf_df.index.map(int)
        conf_df[' '] = conf_df.index
        return [
            div_view.div(
                children=[
                    table_view.create_table(conf_df, class_name='nine columns'),
                    div_view.div(
                        [
                            div_view.div(['Se = {:0.4f}'.format(conf.TP / (conf.TP + conf.FN))]),
                            div_view.div(['+P = {:0.4f}'.format(conf.TP / (conf.TP + conf.FN))])
                        ],
                        class_name='three columns'
                    )
                ],
                class_name='row'
            )
        ]
