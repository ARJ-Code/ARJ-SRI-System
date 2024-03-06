from abc import ABC, abstractmethod
from typing import List, Tuple
import spacy
import gensim

nlp = spacy.load('en_core_web_sm')


class Document(ABC):
    def __init__(self, doc_id: str, title: str, text: str) -> None:
        """
        Initialize a Document with a title and text.

        Args:
            title (str): The title of the document.
            text (str): The content of the document.
        """
        self.doc_id = doc_id
        self.title: str = title
        self.text: str = text


class Corpus(ABC):
    """
    Abstract base class for a corpus of documents.
    """
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
        """
        Abstract base class for a text model.
        """
        self.vocabulary: List[str] = []
        self.relevant_docs = []
        self.non_relevant_docs = []

    def _tokenize_doc(doc) -> List[str]:
        """
        Tokenize a document.

        Args:
            doc: The document to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        return [token.text for token in nlp(
            doc.lower()) if token.is_alpha and not token.is_stop]

    def _lemma(tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            List[str]: A list of lemmatized tokens.
        """
        return [nlp(token)[0].lemma_ for token in tokens]

    def build(self, documents: List[Document]) -> List[Tuple[str, str, List[str]]]:
        """
        Build the model from a list of documents.

        Args:
            documents (List[Document]): List of documents.

        Returns:
            List[Tuple[str, List[str]]]: A list of tuples containing document titles and tokenized text.
        """
        tokenized_docs = [(doc.doc_id, doc.title, Model._tokenize_doc(
            doc.text.lower())) for doc in documents]

        dict_voc = gensim.corpora.Dictionary(
            [doc for _, _, doc in tokenized_docs])
        dict_voc.save('data/vocabulary.dict')

        return tokenized_docs

    def load(self, vocabulary: List[str], relevant_docs, non_relevant_docs):
        """
        Load model data.

        Args:
            documents (List[Document]): List of documents.
            vocabulary (List[str]): List of vocabulary terms.
            relevant_docs: Relevant documents.
            non_relevant_docs: Non-relevant documents.
        """
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
