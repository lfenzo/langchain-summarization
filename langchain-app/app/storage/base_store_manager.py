from abc import ABC, abstractmethod


class BaseStoreManager(ABC):

    @abstractmethod
    def get_summary(self):
        ...

    @abstractmethod
    def store_summary(self, summary: str, metadata: dict, document: bytes):
        ...

    @abstractmethod
    def store_summary_feedback(self):
        ...
