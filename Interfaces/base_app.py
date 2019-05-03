import logging
import os
import sys
import traceback

from dash.dependencies import Input, Output, State


class BaseApp(object):
    def __init__(self, app, name='App', title='Application', loglevel=logging.ERROR):
        self.app = app
        self.name = name
        self.title = title
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        logging.basicConfig(filename=title + '.' + __name__ + '.' + self.__class__.__name__ + '.log', filemode='w',
                            level=loglevel, format='%(asctime)s %(levelname)s:%(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S ')
        self.logger.debug('init app')
        self.store = {}

    def register_callbacks(self, callbacks):
        print('registering {} callback(s) for {}'.format(len(callbacks), self.name))

        for callback_data in callbacks:
            f = self.create_callback(callback_data[0], callback_data[3])
            self.app.callback(output=callback_data[0], inputs=callback_data[1], state=callback_data[2])(f)

    @staticmethod
    def create_callback(output_element, f):
        """creates a callback function"""

        def callback(*input_values):
            print('callback {} fired with :"{}"  output:{}/{}'.format(str(f.__name__),
                                                                      input_values,
                                                                      output_element.component_id,
                                                                      output_element.component_property))
            retval = []
            if input_values is not None and input_values != 'None':
                try:
                    retval = f(*input_values)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = traceback.extract_tb(exc_tb, 1)[0][2]
                    filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print('Callback Exception:', e, exc_type, filename, exc_tb.tb_lineno, fname)
                    print('parameters:', *input_values)
                    traceback.print_tb(exc_tb)

            return retval

        return callback

    def define_callback(self, output, input, func=None, state=None):
        """defines the callback set"""
        return (
            Output(output[0], output[1]),
            [Input(comp_id, attr) for (comp_id, attr) in input],
            [] if state is None else [State(comp_id, attr) for (comp_id, attr) in state],
            self.dummy_callback if func is None else func
        )

    @staticmethod
    def dummy_callback(*input_data):
        print('dummy callback with:', *input_data)
        return []

    def run(self, debug=True):
        self.app.run_server(debug=debug)
