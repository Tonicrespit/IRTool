import dash_core_components as dcc


def create_tab(label: str, value: str):
    return dcc.Tab(label=label, value=value)
