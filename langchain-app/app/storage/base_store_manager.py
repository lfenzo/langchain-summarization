from abc import ABC, abstractmethod


class BaseStoreManager(ABC):

    @abstractmethod
    def store_summary(self, summary: str, metadata: dict, document: bytes):
        pass

    @abstractmethod
    def get_summary_document_id(self):
        pass
