from abc import ABC, abstractmethod
from typing import List


class Document(ABC):
    def __init__(self, title: str, text: str) -> None:
        self.title: str = title
        self.text: str = text


class Corpus(ABC):
    def __init__(self) -> None:
        self.documents: List[Document] = []

    @abstractmethod
    def load(self) -> List[Document]:
        pass


class Model(ABC):
    def __init__(self) -> None:
        self.documents: List[Document] = []

    @abstractmethod
    def build(self, documents: List[Document]):
        pass

    def load(self, documents: List[Document]):
        self.documents = documents
        self._load()

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def query(self, query: str, cant: int) -> List[Document]:
        pass
