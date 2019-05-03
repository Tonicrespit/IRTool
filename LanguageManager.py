import json


class LanguageManager:
    def __init__(self, lang_dir, locale: str = 'en'):
        self.lang_dir = lang_dir
        self.locale = locale

    def change_locale(self, locale: str):
        self.locale = locale

    def translate(self, str_id: str) -> str:
        sub_dir = str_id.rsplit('.', 1)[0]
        sub_dir = sub_dir.replace('.', '/')

        key = str_id.rsplit('.', 1)[1]
        with open(self.lang_dir + '/' + sub_dir + '/{0}.json'.format(self.locale), encoding='utf8') as f:
            strings = json.load(f)
            if key in strings.keys():
                return strings[key]
            else:
                return str_id
