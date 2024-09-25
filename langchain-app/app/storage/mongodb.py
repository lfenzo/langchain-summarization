from typing import Any
from bson import ObjectId
from bson.binary import Binary
from pymongo import MongoClient

from app.models import FeedbackForm
from app.storage import BaseStoreManager


# currently mongodb can only store document of up to 16MB in size
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

    def _get_summary_document_by_id(self, document_id: str) -> dict[str, Any]:
        """Get the summary MongoDB dobument by document_id."""
        collection = self.db[self.collection_name]
        document = collection.find_one({"_id": ObjectId(document_id)})
        return document

    def get_summary(self, **kwargs):
        return self._get_summary_document_by_id(**kwargs)

    async def store_summary(self, _id: str, summary: str, metadata: dict, document: bytes) -> str:
        document = Binary(document) if self.document_can_be_stored(document) else None
        collection = self.db[self.collection_name]

        if not collection.find_one({"_id": _id}):
            summary_entry = {
                "_id": _id,
                "metadata": metadata,
                "summary": summary,
                "original_document_in_bytes": document,
                "feedback": None,
            }
            collection.insert_one(document=summary_entry)
        else:
            # TODO: log here that the current summary was obtained from caching (so we're not
            # inserting it in the database)
            pass

        return _id

    async def store_summary_feedback(self, form: FeedbackForm) -> None:
        collection = self.db[self.collection_name]

        feedback_dict = {
            key: value
            for key, value in form.dict().items() if key != 'document_id'
        }

        update_result = collection.update_one(
            {"_id": ObjectId(form.document_id)},
            {"$set": {"feedback": feedback_dict}}
        )

        if update_result.matched_count == 0:
            raise ValueError(f"Failed to update document with ObjectId '{form.document_id}'")
