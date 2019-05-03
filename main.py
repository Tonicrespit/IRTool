from App import DemoApp
from LanguageManager import LanguageManager
import SettingsParser as st

import dash


if __name__ == '__main__':
    st.init('settings.json')
    lang = LanguageManager(st.get('lang_dir'), st.get('locale'))

    dash_app = dash.Dash(__name__)
    app = DemoApp(dash_app, lang)
    app.run(debug=True)
