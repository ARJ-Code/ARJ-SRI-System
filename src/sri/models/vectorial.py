from ..core import Model, QueryBuilder
from typing import List, Tuple, Dict
import json
import gensim
import numpy as np
from ..utils.methods import sum_vectors, mult_scalar, mean


class Vectorial(Model):
    """
    A class for building and manipulating vectorial models.

    Args:
        query_builders (List[QueryBuilder]): List of query builders.

    Attributes:
        query_builders (List[QueryBuilder]): List of query builders.
        tfidf (gensim.models.TfidfModel): TF-IDF model.
        dictionary (gensim.corpora.Dictionary): Dictionary for vector representation.
        data_build (dict): Dictionary containing vector representations of documents.
        relevant_docs (set): Set of relevant document titles.
        non_relevant_docs (set): Set of non-relevant document titles.

    Methods:
        build_model(tokenized_docs: List[Tuple[str, List[str]]]) -> None:
            Builds the vectorial model from tokenized documents.
        _load() -> None:
            Loads the vectorial data from files.
        __rocchio_algorithm(query) -> None:
            Calculates the Rocchio query with feedback.

    """
    def __init__(self, query_builders: List[QueryBuilder] = []) -> None:
        super().__init__()
        self.query_builders: List[QueryBuilder] = query_builders
        self.data_build: Dict[str, List[float]] = {}

    def build_model(self, tokenized_docs: List[Tuple[str, str, List[str]]]):
        """
        Builds the vectorial model from tokenized documents.

        Args:
            tokenized_docs (List[Tuple[str, List[str]]]): List of tokenized documents.

        Returns:
            None
        """
        tokenized_docs = [(doc_id, t, Model._lemma(doc))
                          for doc_id, t, doc in tokenized_docs]
        dictionary = gensim.corpora.Dictionary(
            [doc for _, _, doc in tokenized_docs])

        corpus = [dictionary.doc2bow(doc) for _, _, doc in tokenized_docs]

        tfidf = gensim.models.TfidfModel(corpus)

        tfidf.save("data/tfidf.model")
        dictionary.save("data/dictionary_vectorial.dict")
        vector_repr = [tfidf[doc] for doc in corpus]

        data_build = {}

        for i in range(len(vector_repr)):
            data_build[tokenized_docs[i][0]
                       ] = tokenized_docs[i][1], vector_repr[i]

        f = open('data/data_build_vectorial.json', 'w')
        json.dump(data_build, f)
        f.close()

    def _load(self):
        """
        Loads the vectorial data from files.

        Returns:
            None
        """
        self.tfidf = gensim.models.TfidfModel.load("data/tfidf.model")
        self.dictionary = gensim.corpora.Dictionary.load(
            "data/dictionary_vectorial.dict")

        f = open('data/data_build_vectorial.json')
        self.data_build = json.load(f)
        f.close()

        self.relevant_docs = set()
        self.non_relevant_docs = set()

    def __rocchio_algorithm(self, query):
        '''
        Rocchio Algorithm

        query: Original query

        return:
        query_rocchio: Modified query
        '''
        a = 1.0  # Weight of the original query
        b = 0.8  # Weight of relevant documents
        c = 0.1  # Weight of non-relevant documents

        # Convertir los documentos relevantes y no relevantes a su representación BoW
        relevant_docs_bow = [self.data_build[doc][1]
                             for doc in self.relevant_docs]
        non_relevant_docs_bow = [self.dictionary.doc2bow(Model._lemma(
            self.data_build[doc][1])) for doc in self.non_relevant_docs]

        mean_relevant = mean(relevant_docs_bow)
        mean_non_relevant = mean(non_relevant_docs_bow)

        # Calculate the modified Rocchio query
        query_rocchio = sum_vectors(sum_vectors(mult_scalar(query, a),  mult_scalar(
            mean_relevant, b)), mult_scalar(mean_non_relevant, c))

        return query_rocchio

    def query(self, query: str, cant: int) -> List[Tuple[str, float]]:
        """
        Executes a vectorial query and returns a list of relevant documents.

        Args:
            query (str): The input query.
            cant (int): The maximum number of relevant documents to return.

        Returns:
            List[Document]: A list of relevant documents.
        """
        query_tokens = Model._tokenize_doc(query)

        for builder in self.query_builders:
            query_tokens = builder.build(query_tokens, self.vocabulary)

        query_bow = self.dictionary.doc2bow(Model._lemma(query_tokens))

        query_tfidf = self.tfidf[query_bow]

        similarities = [(gensim.matutils.cossim(self.__rocchio_algorithm(query_tfidf), v), k, n)
                        for k, (n, v) in self.data_build.items()]

        similarities.sort(reverse=True)

        return [(k, n, v) for v, k, n in similarities[:cant] if v != 0]
