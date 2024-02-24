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
    def __init__(self, corpus: Corpus) -> None:
        self.corpus: Corpus = corpus

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def query(self, query: str) -> List[Document]:
        pass
