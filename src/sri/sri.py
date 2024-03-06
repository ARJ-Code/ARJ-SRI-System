from .core import Corpus, Model
from typing import List, Tuple
from .utils.trie import Trie
import gensim
import json


class SRISystem:
    def __init__(self, corpus: Corpus, models: List[Model]) -> None:
        self.models: List[Model] = models
        self.corpus: Corpus = corpus
        self.trie = Trie()
        self.selected = 0
        self.relevant_docs = []
        self.non_relevant_docs = []

    def build(self, cant_lines=-1):
        self.corpus.load(cant_lines)

        tokenized_docs = self.models[0].build(self.corpus.documents)

        for model in self.models:
            model.build_model(tokenized_docs)

        self.__save_relevant_docs()
        self.__save_non_relevant_docs()

    def change_selected(self, ind: int):
        self.selected = ind

    def load(self, cant_lines: int = -1):
        self.corpus.load(cant_lines)

        dict_voc = gensim.corpora.Dictionary.load(
            "data/vocabulary.dict")
        vocabulary = list(dict_voc.token2id.keys())

        f1 = open('data/relevant_docs.json')
        f2 = open('data/non_relevant_docs.json')

        self.relevant_docs = json.load(f1)
        self.non_relevant_docs = json.load(f2)

        f1.close()
        f2.close()

        for model in self.models:
            model.load(vocabulary,
                       self.relevant_docs, self.non_relevant_docs)

        self.__create_trie(vocabulary)

    def __create_trie(self, words: List[str]):
        for word in words:
            self.trie.insert(word)

    def auto_complete(self, word: str, cant: int = 5) -> List[str]:
        return self.trie.find_closest_words(word, cant)

    def query(self, query: str, cant: int = 10) -> List[Tuple[str, int]]:
        return self.models[self.selected].query(query, cant)

    def add_relevant(self, doc: str):
        if doc in self.relevant_docs:
            return

        self.relevant_docs.append(doc)
        self.__save_relevant_docs()

        self.remove_non_relevant(doc)

    def remove_relevant(self, doc: str):
        self.relevant_docs = [x for x in self.relevant_docs if x != doc]
        self.__save_relevant_docs()

    def add_non_relevant(self, doc: str):
        if doc in self.non_relevant_docs:
            return

        self.non_relevant_docs.append(doc)
        self.__save_non_relevant_docs()

        self.remove_relevant(doc)

    def remove_non_relevant(self, doc: str):
        self.non_relevant_docs = [x for x in self.relevant_docs if x != doc]
        self.__save_non_relevant_docs()

    def __save_relevant_docs(self):
        f1 = open('data/relevant_docs.json', 'w')

        json.dump(self.relevant_docs, f1)

        f1.close()

    def __save_non_relevant_docs(self):
        f2 = open('data/non_relevant_docs.json', 'w')

        json.dump(self.non_relevant_docs, f2)

        f2.close()
