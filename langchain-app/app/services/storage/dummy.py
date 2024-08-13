from app.services.storage.base_storage import BaseStorage


class DummyStorage(BaseStorage):

    def __init__(self):
        print("Storage initialized")

    def store(self, key: str, value):
        return False

    def retrieve(self, key: str):
        return False
