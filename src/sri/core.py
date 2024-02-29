from abc import ABC, abstractmethod
from typing import List
import spacy

nlp = spacy.load('en_core_web_sm')


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


class QueryBuilder(ABC):
    @abstractmethod
    def build(self, tokens: List[str], words: List[str]):
        pass


class Model(ABC):
    def __init__(self, queryBuilders: List[QueryBuilder] = []) -> None:
        self.documents: List[Document] = []
        self.queryBuilders: List[QueryBuilder] = queryBuilders

    def _tokenize_doc(doc) -> List[str]:
        """
        Función que tokeniza un documento y elimina las palabras vacías
        doc: Documento a tokenizar
        """
        return [token.text for token in nlp(
            doc.lower()) if token.is_alpha and not token.is_stop]

    def _lemma(tokens) -> List[str]:
        vocab = spacy.Vocab()
        return [vocab.create_token(token).lemma_ for token in tokens]

    def build(self, documents: List[Document]):
        tokenized_docs = [(doc.title, Model._lemma(Model._tokenize_doc(
            doc.text.lower()))) for doc in documents]

        self._build(tokenized_docs)

    def query(self, query: str, cant: int) -> List[Document]:
        query_tokens = Model._tokenize_doc(query)

        for builder in self.queryBuilders:
            query_tokens = builder.build(query_tokens, self.words())

        self._query(Model._lemma(query_tokens), cant)

    def load(self, documents: List[Document]):
        self.documents = documents
        self._load()

    @abstractmethod
    def _build(self, tokenized_docs: List[List[str]]):
        pass

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def _query(self, query: str, cant: int) -> List[Document]:
        pass

    @abstractmethod
    def words(self) -> List[str]:
        pass
