from app.storage.mongodb import MongoDBStoreManager


class StorageManagerFactory:

    def __init__(self):
        self.storage_managers = {
            'mongodb': MongoDBStoreManager,
        }

    def create(self, manager: str, **kwargs):
        return self.storage_managers[manager](**kwargs)
