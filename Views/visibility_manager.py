def show():
    return {'display': 'block'}


def hide():
    return {'display': 'none'}


def is_hidden(state: dict) -> bool:
    return 'display' in state and state['display'] == 'none'
