import dash_html_components as html
import dash_core_components as dcc

from Interfaces.base_app import BaseApp

from Controllers.data_source_controller import DataSourceController
from Controllers.model_controller import ModelController
from Models.datasource import DataSource


class DemoApp(BaseApp):
    def __init__(self, app, lang):
        # init object
        BaseApp.__init__(self, app)
        self.lang = lang

        # load app layout and initial callbacks
        self.app.layout = self.layout()
        self.init_components()
        self.init_callbacks()
        self.controllers = self.init_controllers()

    def layout(self):
        """defines the web ui layout"""
        self.init_components()
        return html.Div([
            self.header(),

            html.Div([
                html.Div([
                    dcc.Markdown(self.lang.translate('layout.select-data')),
                    dcc.Upload(id='data-file-upload', multiple=False, children=html.Div([
                        self.lang.translate('layout.drag_drop'),
                        html.A(self.lang.translate('layout.select_files'))
                    ], id='upload-container')),
                ], className='six columns'),

                # Train-test split
                html.Div([
                    # Title
                    dcc.Markdown(self.lang.translate('layout.test-train-split')),

                    # Target column
                    html.Div([
                        html.Label(children=[self.lang.translate('layout.select-target')], className='three columns'),
                        html.Div([
                            dcc.Dropdown(
                                options=[],
                                id='data-test-train-split-select-dropdown'
                            ),
                        ], className='nine columns')
                    ], className='row'),

                    # Train size
                    html.Div([
                        html.Label(children=[self.lang.translate('layout.train-size')], className='three columns'),
                        dcc.Input(id='data-test-train-split-train-size', className='nine columns', type='text')
                    ], className='row'),

                    # Random state seed
                    html.Div([
                        html.Label(children=[self.lang.translate('layout.random-state')], className='three columns'),
                        dcc.Input(id='data-test-train-split-random-state', className='nine columns', type='text')
                    ], className='row'),

                    # Stratify
                    html.Div([
                        html.Label(children=[self.lang.translate('layout.stratify')], className='three columns'),
                        dcc.Checklist(id='data-test-train-split-stratify',
                                      options=[{'label': '', 'value': 'stratify'}],
                                      values=[],
                                      className='nine columns')
                    ], className='row'),

                    # Split button
                    html.Div([
                        html.Label('', className='three columns empty-space'),
                        html.Div([
                            html.Button(self.lang.translate('layout.submit-train-test-split'),
                                        id='data-test-train-split-submit', className='nine columns')
                        ], className='nine columns')
                    ], className='row'),
                ], id='data-test-train-split-container', className='six columns', style={'display': 'none'}),
            ], className='row'),

            html.Div(id='data-summary-container', children=[
                # Preview data
                dcc.Markdown(self.lang.translate('layout.text1')),
                dcc.Tabs(id='data-summary-tabs', value='no-tab', children=[
                    dcc.Tab(label='Data preview', value='tabs-1-preview'),
                    dcc.Tab(label='Data summary', value='tabs-1-summary'),
                ]),
                html.Div(id='data-summary-tab-content'),

                html.Div([
                    html.Div([
                        # Select useful columns
                        dcc.Markdown(self.lang.translate('layout.text2')),
                        dcc.Checklist(
                            options=[],
                            values=[],
                            id='columns-to-use-checklist'
                        ),
                    ], className='six columns'),
                    html.Div([
                        # Select target value
                        dcc.Markdown(self.lang.translate('layout.text4')),
                        dcc.Dropdown(
                            options=[],
                            id='target-value-dropdown'
                        ),
                    ], className='six columns'),
                ], className='row'),

                # Categorize rows
                html.Div(id='data-categorize-parent', children=[
                    dcc.Markdown(self.lang.translate('layout.text5')),
                    html.Div([], id='categorize-tabs-trigger', style={'display': 'none'}),
                    dcc.Tabs(id='data-categorize-tabs', value='no-tab'),
                    html.Div(id='data-categorize-tab-content', children=[
                        dcc.Slider(min=3, max=13, step=1, value=4, marks={i+1: i for i in range(2, 12)},
                                   id='data-ncategories-slider'),
                        dcc.RangeSlider(min=0, max=12, value=[], dots=False, step=0.01, updatemode='drag',
                                        allowCross=False, id='data-quantiles-slider'),

                        html.Div(id='categorized-column-preview'),
                    ]),
                    html.Button(self.lang.translate('layout.categorize_button_text'), id='categorize-data-button')
                ], style={'display': 'none'}),

                # Do stuff with IRT models
                html.Div(id='model-output-container', children=[
                    dcc.Markdown(self.lang.translate('layout.select-model')),
                    html.Div([
                        html.Div([
                            dcc.Dropdown(id='choose-model-dropdown', options=[]),
                        ], className='six columns'),
                        html.Div([
                            html.Button(self.lang.translate('layout.test-model-text'), id='test-model-button'),
                        ], className='six columns'),
                    ], className='row'),

                    html.Div(id='model-output', children=[
                        dcc.Markdown(self.lang.translate('layout.model-output')),
                        dcc.Tabs(id='model-output-tabs', value='default'),
                        html.Div(id='model-output-pane', className="row"),
                        dcc.Markdown(self.lang.translate('layout.prediction')),
                        html.Div(className='row', children=[
                            html.Div(id='model-output-roc', className='nine columns'),
                            html.Div(id='model-output-threshold-container', className='three columns', children=[
                                html.Div(className='row', children=[
                                    dcc.Markdown(self.lang.translate('layout.choose-threshold')),
                                    dcc.Input(id='model-output-threshold', type='text'),
                                ]),
                                html.Button(self.lang.translate('layout.choose-threshold-button'),
                                            id='choose-threshold-button'),
                            ]),
                        ])
                    ], style={'display': 'none'}),
                    html.Div(id='test-results'),
                ], style={'display': 'none'}),
            ], style={'display': 'none'}),

            self.footer(),
        ])

    def init_components(self):
        """define custom ui components"""
        pass

    def init_callbacks(self):
        callbacks = []
        self.register_callbacks(callbacks)

    def set_callbacks(self, callbacks):
        self.register_callbacks(callbacks)

    def set_callback(self, callback):
        self.register_callbacks([callback])

    def add_controller(self, name, controller):
        if name not in self.controllers.keys():
            self.controllers[name] = controller

    def init_controllers(self):
        ds_controller = DataSourceController(self)
        irt_controller = ModelController(self)

        return {
            'datasource': ds_controller,
            'irt_model': irt_controller
        }

    def get_data(self) -> DataSource:
        return self.controllers['datasource'].data

    def header(self):
        return html.Div([
            dcc.Markdown(self.lang.translate('layout.title')),
            dcc.Markdown(self.lang.translate('layout.description')),
        ])

    def footer(self):
        return html.Footer([
            html.Div([
                # dcc.Markdown(self.lang.translate('layout.footer'))
            ])
        ])
