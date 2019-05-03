class BaseView:
    def __init__(self, container_id: str, app: object):
        self.container_id = container_id
        self.app = app

    def init(self):
        pass

    def create_layout(self, state):
        pass
