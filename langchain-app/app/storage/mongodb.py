from bson import ObjectId
from bson.binary import Binary
from pymongo import MongoClient

from app.storage.base_store_manager import BaseStoreManager


MAX_DOCUMENT_SIZE_IN_BYTES = 16_793_598  # from pymongo error message (~16MB)


class MongoDBStoreManager(BaseStoreManager):

    def __init__(self, user: str, password: str, database_name: str, collection_name: str):
        connection_string = self.get_connection_string(user=user, password=password)
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = MongoClient(connection_string)
        self.db = self.client[self.database_name]

    def get_connection_string(self, user: str, password: str) -> str:
        return f"mongodb://{user}:{password}@mongodb:27017/"

    def document_can_be_stored(self, document: bytes) -> bool:
        return len(document) <= MAX_DOCUMENT_SIZE_IN_BYTES

    async def store_summary(self, summary: str, metadata: dict, document: bytes) -> str:
        collection = self.db[self.collection_name]
        # currently mongodb can only store document of up to 16MB in size
        document = Binary(document) if self.document_can_be_stored(document) else None

        document_id = collection.insert_one({
            "metadata": metadata,
            "summary": summary,
            "original_document_in_bytes": document,
        }).inserted_id
        return str(document_id)

    def get_summary_document_id(self, document_id: str) -> bytes:
        collection = self.db[self.database_name]
        document = collection.find_one({"_id": ObjectId(document_id)})
        return document['data']
