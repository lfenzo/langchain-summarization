from abc import ABC, abstractmethod


class BaseStoreManager(ABC):

    @abstractmethod
    def get_summary(self):
        pass

    @abstractmethod
    def store_summary(self, summary: str, metadata: dict, document: bytes):
        pass

    @abstractmethod
    def store_summary_feedback(self):
        pass
