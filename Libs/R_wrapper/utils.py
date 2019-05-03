from rpy2.rinterface import RNULLType
from rpy2.robjects.vectors import *
from rpy2.robjects.functions import SignatureTranslatedFunction
import pandas as pd
import numpy as np


def list_to_py(l):
    data = {}
    for key in l.names:
        if type(l.rx2(key)) is not RNULLType:
            data[key] = to_py(l.rx2(key))
    return data


def dict_to_py(data):
    dictionary = {}
    for key in data:
        if type(data[key]) is not RNULLType:
            dictionary[key] = to_py(data[key])
    return data


def to_py(v):
    if type(v) in [FloatVector, IntVector] and type(v.names) is RNULLType:
        return np.array(v)
    elif type(v) in [BoolVector]:
        return np.array(v, dtype=bool)
    elif type(v) in [StrVector]:
        return np.array(v, dtype=str)
    elif type(v) in [ListVector]:
        return list_to_py(v)
    elif type(v) in [Matrix] and type(v.names) is RNULLType:
        return np.matrix(v)
    elif type(v) in [DataFrame] or (type(v) in [FloatVector, IntVector, Matrix] and type(v.names) is not RNULLType):
        if type(v) in [Matrix]:
            names = v.names[1]
        else:
            names = v.names

        if type(v) in [Matrix]:
            data = np.matrix(v)
        else:
            data = np.array(v)

        if type(v) in [Matrix]:
            pass
        elif len(data.shape) == 1:
            data = data.reshape(-1, len(data))
        else:
            data = data.transpose()

        df = pd.DataFrame(data, columns=names)
        if hasattr(v[0], 'names') and type(v.names) is RNULLType:
            df.index = v[0].names
        return df
    elif type(v) in [SignatureTranslatedFunction]:
        return []
    else:
        return 'Method not implemented for type {}'.format(type(v))
