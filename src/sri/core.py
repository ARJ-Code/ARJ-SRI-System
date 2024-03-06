from abc import ABC, abstractmethod
from typing import List, Tuple
import spacy
import gensim

nlp = spacy.load('en_core_web_sm')


class Document(ABC):
    def __init__(self, doc_id: str, title: str, text: str) -> None:
        self.doc_id = doc_id
        self.title: str = title
        self.text: str = text


class Corpus(ABC):
    def __init__(self) -> None:
        self.documents: List[Document] = []

    @abstractmethod
    def load(self, cant: int = -1) -> List[Document]:
        pass

    def get_qrels(self) -> List:
        return []

    def get_queries(self) -> List:
        return []


class QueryBuilder(ABC):
    @abstractmethod
    def build(self, tokens: List[str], words: List[str]) -> List[str]:
        pass


class Model(ABC):
    def __init__(self) -> None:
        self.vocabulary: List[str] = []
        self.relevant_docs = []
        self.non_relevant_docs = []

    def _tokenize_doc(doc) -> List[str]:
        """
        Función que tokeniza un documento y elimina las palabras vacías
        doc: Documento a tokenizar
        """
        return [token.text for token in nlp(
            doc.lower()) if token.is_alpha and not token.is_stop]

    def _lemma(tokens: List[str]) -> List[str]:
        return [nlp(token)[0].lemma_ for token in tokens]

    def build(self, documents: List[Document]) -> List[Tuple[str, str, List[str]]]:
        tokenized_docs = [(doc.doc_id, doc.title, Model._tokenize_doc(
            doc.text.lower())) for doc in documents]

        dict_voc = gensim.corpora.Dictionary(
            [doc for _, _, doc in tokenized_docs])
        dict_voc.save('data/vocabulary.dict')

        return tokenized_docs

    def load(self, vocabulary: List[str], relevant_docs, non_relevant_docs):
        self.vocabulary = vocabulary
        self.relevant_docs = relevant_docs
        self.non_relevant_docs = non_relevant_docs
        self._load()

    @abstractmethod
    def query(self, query: str, cant: int) -> List[Tuple[str, str, float]]:
        self._query(Model._lemma(query), cant)

    @abstractmethod
    def build_model(self, tokenized_docs: List[Tuple[str, str, List[str]]]):
        pass

    @abstractmethod
    def _load(self):
        pass
