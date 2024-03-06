from ..core import Model, QueryBuilder
from typing import List, Tuple, Dict
import json
import gensim
import numpy as np
from gensim.models import LsiModel

import numpy as np
from ..utils.methods import sum_vectors, mult_scalar, mean


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class LSI(Model):
        
    def __init__(self, query_builders: List[QueryBuilder] = []) -> None:
        """
        Initialize an LSI model.

        Args:
            query_builders: List of query builders.
        """
        super().__init__()
        self.query_builders: List[QueryBuilder] = query_builders
        self.data_build: Dict[str, List[float]] = {}

    def build_model(self, tokenized_docs: List[Tuple[str, str, List[str]]]):
        """
        Build the LSI model.

        Args:
            A list of tuples containing document information.
        """

        tokenized_docs = [(doc_id, t, Model._lemma(doc))
                          for doc_id, t, doc in tokenized_docs]
        dictionary = gensim.corpora.Dictionary(
            [doc for _, _, doc in tokenized_docs])

        corpus = [dictionary.doc2bow(doc) for _, _, doc in tokenized_docs]

        num_topics = 100  
        lsi = LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

        lsi.save("data/lsi.model")
        dictionary.save("data/dictionary_lsi.dict")

        vector_repr = [lsi[doc] for doc in corpus]

        data_build = {}

        for i in range(len(vector_repr)):
            data_build[tokenized_docs[i][0]
                       ] = tokenized_docs[i][1], vector_repr[i]

        with open('data/data_build_lsi.json', 'w') as f:
            json.dump(data_build, f, cls=NpEncoder)

    def _load(self):
        """
        Load the LSI model.
        """

        self.lsi = LsiModel.load("data/lsi.model")
        self.dictionary = gensim.corpora.Dictionary.load(
            "data/dictionary_lsi.dict")

        with open('data/data_build_lsi.json') as f:
            self.data_build = json.load(f)

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

    def query(self, query: str, cant: int) -> List[Document]:
        """
        Executes a LSI query and returns a list of relevant documents.

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

        query_lsi = self.lsi[query_bow]

        similarities = [(gensim.matutils.cossim(self.__rocchio_algorithm(query_lsi), v), k, n)
                        for k, (n, v) in self.data_build.items()]

        similarities.sort(reverse=True)

        return [(k, n, v) for v, k, n in similarities[:cant] if v != 0]
