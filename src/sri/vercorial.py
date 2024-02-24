from .core import Model, Corpus, Document
from typing import List
import json
import spacy
import gensim
import logging
import numpy as np
from gensim.matutils import corpus2dense

nlp = spacy.load('en_core_web_sm')
CANT = 10


class Vectorial(Model):
    def __init__(self, corpus: Corpus) -> None:
        super.__init__(corpus)

    def __tokenize_doc(doc):
        """
        Función que tokeniza un documento y elimina las palabras vacías
        doc: Documento a tokenizar
        """

        return [token.lemma_ for token in nlp(
            doc.lower()) if token.is_alpha and not token.is_stop]

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

    def build(self):
        tokenized_docs = [Vectorial.__tokenize_doc(
            doc.text.lower()) for doc in self.corpus.documents]

        dictionary = gensim.corpora.Dictionary(tokenized_docs)

        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

        tfidf = gensim.models.TfidfModel(corpus)

        logging.info('Save data')
        tfidf.save("data/tfidf.model")
        dictionary.save("data/dictionary.dict")
        vector_repr = [tfidf[doc] for doc in corpus]

        data_build = {}

        for i in range(len(vector_repr)):
            data_build[self.corpus.documents[i].title] = vector_repr[i]

        f = open('data/data_build.json', 'w')
        json.dump(data_build, f)
        f.close()

    def load(self):

        # Cargar el modelo TF-IDF y el diccionario
        self.tfidf = gensim.models.TfidfModel.load("data/tfidf.model.news")
        self.dictionary = gensim.corpora.Dictionary.load(
            "data/dictionary.dict.news")

        f = open('data/data_build.json')
        self.data_build = json.load(f)
        f.close()

    def query(self, query: str) -> List[Document]:
        query_tokens = Vectorial.__tokenize_doc(query)

        # Convertir la consulta en su representación BoW
        query_bow = self.dictionary.doc2bow(query_tokens)

        # Calcular la representación TF-IDF de la consulta
        query_tfidf = self.tfidf[query_bow]

        # Calcular la similitud entre la consulta y cada documento en el corpus
        similarities = [Vectorial.__cosine_similarity(query_tfidf, self.data_build[doc.title], self.dictionary)
                        for doc in self.corpus.documents]

        # Ordenar las noticias por similitud y seleccionar las más relevantes
        top_n_indices = np.argsort(similarities)[-CANT:]

        top_n = [self.corpus.documents[ind]
                 for ind in top_n_indices if similarities[ind] != 0]
        top_n.reverse()
