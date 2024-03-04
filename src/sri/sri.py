from .core import Corpus, Model, Document
from typing import List
from .trie import Trie


class SRISystem:
    def __init__(self, corpus: Corpus, model: Model) -> None:
        self.model: Model = model
        self.corpus: Corpus = corpus
        self.trie = Trie()

    def build(self):
        self.corpus.load()
        self.model.build(self.corpus.documents)

    def load(self):
        self.corpus.load()
        self.model.load(self.corpus.documents)

        self.__create_trie(self.model.vocabulary)

    def __create_trie(self, words: List[str]):
        for word in words:
            self.trie.insert(word)

    def auto_complete(self, word: str, cant: int = 5) -> List[str]:
        return self.trie.find_closest_words(word, cant)

    def query(self, query: str, cant: int = 10) -> List[Document]:
        return self.model.query(query, cant)
