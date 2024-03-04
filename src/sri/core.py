from abc import ABC, abstractmethod
from typing import List
import spacy
import gensim

nlp = spacy.load('en_core_web_sm')


class Document(ABC):
    def __init__(self, title: str, text: str) -> None:
        self.title: str = title
        self.text: str = text


class Corpus(ABC):
    def __init__(self) -> None:
        self.documents: List[Document] = []

    @abstractmethod
    def load(self, cant: int = -1) -> List[Document]:
        pass


class QueryBuilder(ABC):
    @abstractmethod
    def build(self, tokens: List[str], words: List[str]) -> List[str]:
        pass


class Model(ABC):
    def __init__(self) -> None:
        self.documents: List[Document] = []
        self.vocabulary: List[str] = []

    def _tokenize_doc(doc) -> List[str]:
        """
        Función que tokeniza un documento y elimina las palabras vacías
        doc: Documento a tokenizar
        """
        return [token.text for token in nlp(
            doc.lower()) if token.is_alpha and not token.is_stop]

    def _lemma(tokens: List[str]) -> List[str]:
        return [nlp(token)[0].lemma_ for token in tokens]

    def build(self, documents: List[Document]):
        tokenized_docs = [(doc.title, Model._tokenize_doc(
            doc.text.lower())) for doc in documents]

        dict_voc = gensim.corpora.Dictionary(
            [doc for _, doc in tokenized_docs])
        dict_voc.save('data/vocabulary.dict')

        self._build(tokenized_docs)

    def load(self, documents: List[Document]):
        dict_voc = gensim.corpora.Dictionary.load(
            "data/vocabulary.dict")
        self.vocabulary = list(dict_voc.token2id.keys())

        self.documents = documents
        self._load()

    @abstractmethod
    def query(self, query: str, cant: int) -> List[Document]:
        self._query(Model._lemma(query), cant)

    @abstractmethod
    def _build(self, tokenized_docs: List[List[str]]):
        pass

    @abstractmethod
    def _load(self):
        pass
