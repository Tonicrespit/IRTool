from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
import pandas as pd
from io import StringIO

_ltm = importr('ltm')

_data = {
    'LSAT': data(_ltm).fetch('LSAT')['LSAT'],
    'Environment': data(_ltm).fetch('Environment')['Environment'],
    'Science': data(_ltm).fetch('Science')['Science']
}


def _wrapper():
    pandas2ri.activate()


def data(key: str) -> object:
    _wrapper()
    return _data[key]


def rasch(data, **kwargs):
    _wrapper()
    return _ltm.rasch(data, **kwargs)


def tpm(data, **kwargs):
    _wrapper()
    return _ltm.tpm(data, **kwargs)


def grm(data, **kwargs):
    _wrapper()
    return _ltm.grm(data, **kwargs)


def plot(model_robject, model_name, **kwargs):
    if model_name not in ['grm', 'ltm', 'rasch', 'tpm']:
        return []
    _wrapper()
    f = getattr(_ltm, 'plot_'+model_name)
    return f(model_robject, **kwargs)


def factor_scores(model_robject, *args, **kwargs):
    _wrapper()
    return _ltm.factor_scores(model_robject, *args, **kwargs)


def get(o, type):
    if type == 'df':
        return pd.read_csv(StringIO(str(o)), sep='\s+')
