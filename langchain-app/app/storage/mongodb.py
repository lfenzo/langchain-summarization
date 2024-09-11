from bson import ObjectId
from bson.binary import Binary
from pymongo import MongoClient

from app.models.feedback import FeedbackForm
from app.storage.base_store_manager import BaseStoreManager


MAX_DOCUMENT_SIZE_IN_BYTES = 16_793_598  # obtained from pymongo error message (~16MB)


class MongoDBStoreManager(BaseStoreManager):

    def __init__(
        self,
        user: str = 'root',
        password: str = 'password',
        port: str = '27017',
        database_name: str = 'summary_database',
        collection_name: str = 'summaries',
    ) -> None:
        connection_string = self.get_connection_string(user=user, password=password, port=port)
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = MongoClient(connection_string)
        self.db = self.client[self.database_name]

    def get_connection_string(self, user: str, password: str, port: str) -> str:
        return f"mongodb://{user}:{password}@mongodb:{port}/"

    def document_can_be_stored(self, document: bytes) -> bool:
        return len(document) <= MAX_DOCUMENT_SIZE_IN_BYTES

    def get_summary(self, **kwargs):
        return self._get_summary_by_document_id(**kwargs)

    def _get_summary_by_document_id(self, document_id: str) -> bytes:
        collection = self.db[self.database_name]
        document = collection.find_one({"_id": ObjectId(document_id)})
        return document['data']

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

    async def store_summary_feedback(self, form: FeedbackForm) -> str:
        document = self._get_summary_by_document_id(document_id=form.document_id)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
        print(document)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
