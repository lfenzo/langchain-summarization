from app.storage.mongodb import MongoDBStoreManager


class StoreManagerFactory:

    def __init__(self):
        self.store_managers = {
            'mongodb': MongoDBStoreManager,
        }

    def create(self, store_manager: str, **kwargs):
        return self.store_managers[store_manager](**kwargs)
