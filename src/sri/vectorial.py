from .core import Model, QueryBuilder, Document
from typing import List, Tuple
import json
import gensim
import numpy as np
from gensim.matutils import corpus2dense


class Vectorial(Model):
    def __init__(self, query_builders: List[QueryBuilder] = []) -> None:
        super().__init__()
        self.query_builders: List[QueryBuilder] = query_builders

    def __dense_vect(vect, dictionary):
        """
        Función que convierte un vector disperso en uno denso
        vect: Vector disperso
        dictionary: Diccionario que mapea las palabras a su índice
        """

        return corpus2dense([vect], len(dictionary)).flatten()

    def __cosine_similarity(vec_1, vec_2, dictionary):
        """
        Función que calcula la similitud coseno entre dos vectores
        vec_1: Primer vector
        vec_2: Segundo vector
        dictionary: Diccionario que mapea las palabras a su índice
        """

        vec_1, vec_2 = Vectorial.__dense_vect(
            vec_1, dictionary), Vectorial.__dense_vect(vec_2, dictionary)

        v = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
        if v == 0:
            return 0

        return np.dot(vec_1, vec_2) / v

    def _build(self, tokenized_docs: List[Tuple[str, List[str]]]):
        tokenized_docs = [(t, Model._lemma(doc)) for t, doc in tokenized_docs]
        dictionary = gensim.corpora.Dictionary(
            [doc for _, doc in tokenized_docs])

        corpus = [dictionary.doc2bow(doc) for _, doc in tokenized_docs]

        tfidf = gensim.models.TfidfModel(corpus)

        tfidf.save("data/tfidf.model")
        dictionary.save("data/dictionary.dict")
        vector_repr = [tfidf[doc] for doc in corpus]

        data_build = {}

        for i in range(len(vector_repr)):
            data_build[tokenized_docs[i][0]] = vector_repr[i]

        f = open('data/data_build.json', 'w')
        json.dump(data_build, f)
        f.close()

    def _load(self):

        # Cargar el modelo TF-IDF y el diccionario
        self.tfidf = gensim.models.TfidfModel.load("data/tfidf.model")
        self.dictionary = gensim.corpora.Dictionary.load(
            "data/dictionary.dict")

        f = open('data/data_build.json')
        self.data_build = json.load(f)
        f.close()

    def query(self, query: str, cant: int) -> List[Document]:
        query_tokens = Model._tokenize_doc(query)

        for builder in self.query_builders:
            query_tokens = builder.build(query_tokens, self.vocabulary)

        # Convertir la consulta en su representación BoW
        query_bow = self.dictionary.doc2bow(Model._lemma(query_tokens))

        # Calcular la representación TF-IDF de la consulta
        query_tfidf = self.tfidf[query_bow]

        # Calcular la similitud entre la consulta y cada documento en el corpus
        similarities = [Vectorial.__cosine_similarity(query_tfidf, self.data_build[doc.title], self.dictionary)
                        for doc in self.documents]

        # Ordenar las noticias por similitud y seleccionar las más relevantes
        top_n_indices = np.argsort(similarities)[-cant:]

        top_n = [self.documents[ind]
                 for ind in top_n_indices if similarities[ind] != 0]
        top_n.reverse()

        return top_n
