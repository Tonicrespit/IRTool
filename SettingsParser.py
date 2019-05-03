import json


_settings = None


def init(settings_file: str = 'settings.json'):
    global _settings
    with open(settings_file) as f:
        _settings = json.load(f)


def get(key: str):
    global _settings
    if _settings is None:
        init()
    return _settings[key]
