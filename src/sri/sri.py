from .core import Corpus, Model, Document
from typing import List
from .utils.trie import Trie
import gensim
import json


class SRISystem:
    def __init__(self, corpus: Corpus, models: List[Model]) -> None:
        """
        Initializes an SRISystem instance.

        Args:
            corpus (Corpus): The corpus containing documents.
            models (List[Model]): List of models used in the system.
        """
        self.models: List[Model] = models
        self.corpus: Corpus = corpus
        self.trie = Trie()
        self.selected = 0
        self.relevant_docs = []
        self.non_relevant_docs = []

    def build(self, cant_lines=-1):
        """
        Builds the SRISystem by loading the corpus and constructing models.

        Args:
            cant_lines (int, optional): Number of lines to load from the corpus. Defaults to -1 (load all).
        """
        self.corpus.load(cant_lines)

        tokenized_docs = self.models[0].build(self.corpus.documents)

        for model in self.models:
            model.build_model(tokenized_docs)

        self.__save_relevant_docs()
        self.__save_non_relevant_docs()

    def change_selected(self, ind: int):
        """
        Changes the selected model index.

        Args:
            ind (int): Index of the selected model.
        """
        self.selected = ind

    def load(self, cant_lines: int = -1):
        """
        Loads the corpus and relevant/non-relevant documents.

        Args:
            cant_lines (int, optional): Number of lines to load from the corpus. Defaults to -1 (load all).
        """
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
            model.load(self.corpus.documents, vocabulary,
                       self.relevant_docs, self.non_relevant_docs)

        self.__create_trie(vocabulary)

    def __create_trie(self, words: List[str]):
        """
        Creates a trie from a list of words.

        Args:
            words (List[str]): List of words to insert into the trie.
        """
        for word in words:
            self.trie.insert(word)

    def auto_complete(self, word: str, cant: int = 5) -> List[str]:
        """
        Provides auto-completions for a given word.

        Args:
            word (str): The input word.
            cant (int, optional): Number of completions to return. Defaults to 5.

        Returns:
            List[str]: List of closest words.
        """
        return self.trie.find_closest_words(word, cant)

    def query(self, query: str, cant: int = 10) -> List[Document]:
        """
        Executes a query using the selected model.

        Args:
            query (str): The query string.
            cant (int, optional): Number of relevant documents to return. Defaults to 10.

        Returns:
            List[Document]: List of relevant documents.
        """
        return self.models[self.selected].query(query, cant)

    def add_relevant(self, doc: str):
        """
        Adds a document to the relevant documents list.

        Args:
            doc (str): The document to add.
        """
        if doc in self.relevant_docs:
            return

        self.relevant_docs.append(doc)
        self.__save_relevant_docs()

        self.remove_non_relevant(doc)

    def remove_relevant(self, doc: str):
        """
        Removes a document from the relevant documents list.

        Args:
            doc (str): The document to remove.
        """
        self.relevant_docs = [x for x in self.relevant_docs if x != doc]
        self.__save_relevant_docs()

    def add_non_relevant(self, doc: str):
        """
        Adds a document to the non-relevant documents list.

        Args:
            doc (str): The document to add.
        """
        if doc in self.non_relevant_docs:
            return

        self.non_relevant_docs.append(doc)
        self.__save_non_relevant_docs()

        self.remove_relevant(doc)

    def remove_non_relevant(self, doc: str):
        """
        Removes a document from the non-relevant documents list.

        Args:
            doc (str): The document to remove.
        """
        self.non_relevant_docs = [x for x in self.relevant_docs if x != doc]
        self.__save_non_relevant_docs()

    def __save_relevant_docs(self):
        """
        Saves the relevant documents to a JSON file.
        """
        f1 = open('data/relevant_docs.json', 'w')

        json.dump(self.relevant_docs, f1)

        f1.close()

    def __save_non_relevant_docs(self):
        """
        Saves the non-relevant documents to a JSON file.
        """
        f2 = open('data/non_relevant_docs.json', 'w')

        json.dump(self.non_relevant_docs, f2)

        f2.close()
