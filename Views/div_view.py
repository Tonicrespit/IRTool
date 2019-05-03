import dash_core_components as dcc
import dash_html_components as html


def div(children, class_name=None):
    return html.Div(children=children, className=class_name)
