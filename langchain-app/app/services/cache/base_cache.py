from abc import ABC, abstractmethod


class BaseCache(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set(self, key: str, value):
        pass

    @abstractmethod
    def get(eslf, key: str):
        pass
