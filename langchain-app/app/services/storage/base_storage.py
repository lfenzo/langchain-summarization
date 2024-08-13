from abc import ABC, abstractmethod


class BaseStorage(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def store(self, key: str, data: bytes):
        pass

    @abstractmethod
    def retrieve(eslf, key: str) -> bytes:
        pass
