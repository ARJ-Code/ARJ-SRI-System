from ..core import Model, QueryBuilder, Document
from typing import List, Tuple
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
        super().__init__()
        self.query_builders: List[QueryBuilder] = query_builders

    def build_model(self, tokenized_docs: List[Tuple[str, List[str]]]):
        tokenized_docs = [(t, Model._lemma(doc)) for t, doc in tokenized_docs]
        dictionary = gensim.corpora.Dictionary(
            [doc for _, doc in tokenized_docs])

        corpus = [dictionary.doc2bow(doc) for _, doc in tokenized_docs]

        # Aplicar LSI
        num_topics = 100  # Número de temas que quieres extraer
        lsi = LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

        lsi.save("data/lsi.model")
        dictionary.save("data/dictionary_lsi.dict")

        # Guardar la representación LSI de cada documento
        vector_repr = [lsi[doc] for doc in corpus]

        data_build = {}

        for i in range(len(vector_repr)):
            data_build[tokenized_docs[i][0]] = vector_repr[i]

        with open('data/data_build_lsi.json', 'w') as f:
            json.dump(data_build, f, cls=NpEncoder)

    def _load(self):
        # Cargar el modelo LSI y el diccionario
        self.lsi = LsiModel.load("data/lsi.model")
        self.dictionary = gensim.corpora.Dictionary.load(
            "data/dictionary_lsi.dict")

        with open('data/data_build_lsi.json') as f:
            self.data_build = json.load(f)

        self.relevant_docs = set()
        self.non_relevant_docs = set()

    def __rocchio_algorithm(self, query):
        # Calcular la consulta Rocchio con retroalimentación
        a = 1.0  # Peso de la consulta original
        b = 0.8  # Peso de los documentos relevantes
        c = 0.1  # Peso de los documentos no relevantes

        # Convertir los documentos relevantes y no relevantes a su representación BoW
        relevant_docs_bow = [self.data_build[doc]
                             for doc in self.relevant_docs]
        non_relevant_docs_bow = [self.dictionary.doc2bow(Model._lemma(
            self.data_build[doc])) for doc in self.non_relevant_docs]

        # Calcular la media de los documentos relevantes y no relevantes
        mean_relevant = mean(relevant_docs_bow)
        mean_non_relevant = mean(non_relevant_docs_bow)

        # Calcular la consulta Rocchio modificada
        query_rocchio = sum_vectors(sum_vectors(mult_scalar(query, a),  mult_scalar(
            mean_relevant, b)), mult_scalar(mean_non_relevant, c))
        
        return query_rocchio

    def query(self, query: str, cant: int) -> List[Document]:
        query_tokens = Model._tokenize_doc(query)

        for builder in self.query_builders:
            query_tokens = builder.build(query_tokens, self.vocabulary)

        # Convertir la consulta en su representación BoW
        query_bow = self.dictionary.doc2bow(Model._lemma(query_tokens))

        # Convertir la consulta en su representación LSI
        query_lsi = self.lsi[query_bow]
        
        similarities = [gensim.matutils.cossim(self.__rocchio_algorithm(query_lsi), self.data_build[doc.title])
                        for doc in self.documents]

        # Ordenar los documentos por similitud y seleccionar las más relevantes
        top_n_indices = np.argsort(similarities)[-cant:]

        top_n = [self.documents[ind]
                 for ind in top_n_indices if similarities[ind] != 0]
        top_n.reverse()

        return top_n
