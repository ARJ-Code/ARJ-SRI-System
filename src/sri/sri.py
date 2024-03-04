from .core import Corpus, Model, Document
from typing import List
from .trie import Trie
import gensim


class SRISystem:
    def __init__(self, corpus: Corpus, models: List[Model]) -> None:
        self.models: List[Model] = models
        self.corpus: Corpus = corpus
        self.trie = Trie()
        self.selected = 0

    def build(self, cant_lines=-1):
        self.corpus.load(cant_lines)

        tokenized_docs = self.models[0].build(self.corpus.documents)

        for model in self.models:
            model.build_model(tokenized_docs)

    def change_selected(self, ind: int):
        self.selected = ind

    def load(self, cant_lines: int = -1):
        self.corpus.load(cant_lines)

        dict_voc = gensim.corpora.Dictionary.load(
            "data/vocabulary.dict")
        vocabulary = list(dict_voc.token2id.keys())

        for model in self.models:
            model.load(self.corpus.documents, vocabulary)

        self.__create_trie(vocabulary)

    def __create_trie(self, words: List[str]):
        for word in words:
            self.trie.insert(word)

    def auto_complete(self, word: str, cant: int = 5) -> List[str]:
        return self.trie.find_closest_words(word, cant)

    def query(self, query: str, cant: int = 10) -> List[Document]:
        return self.models[self.selected].query(query, cant)
