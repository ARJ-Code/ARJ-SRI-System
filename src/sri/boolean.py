from .core import Model, QueryBuilder, Document
from typing import List, Tuple
import json
import gensim
import numpy as np
from gensim.matutils import corpus2dense
from sympy import sympify, to_dnf
import spacy


nlp = spacy.load('en_core_web_sm')


class Boolean (Model):
    def __init__(self, query_builders: List[QueryBuilder] = []) -> None:
        super().__init__()
        self.query_builders: List[QueryBuilder] = query_builders

    def _build(self, tokenized_docs: List[List[str]]):
        tokenized_docs = [(t, doc) for t, doc in tokenized_docs]

        dictionary = gensim.corpora.Dictionary(
            [doc for _, doc in tokenized_docs])

        corpus = [dictionary.doc2bow(doc) for _,doc in tokenized_docs]
        dictionary.save("data/dictionary.dict")

        data_build = {}

        # duda
        for doc in corpus:
            data_build[tokenized_docs[doc][0]] = corpus[doc]

        f = open('data/data_build.json', 'w')
        json.dump(data_build, f)
        f.close()

    # duda
    def _load(self):
        self.dictionary = gensim.corpora.Dictionary.load(
            "data/dictionary.dict")

        f = open('data/data_build.json')
        data_build = json.load(f)
        f.close()
        return data_build

    def tokenize_query(self, query: str) -> List[str]:
        exceptions = ["and", "or", "not", "(", ")", "&", "|", "!"]
        query = [token.lemma_ for token in nlp(query.lower(
        )) if token.lemma in exceptions or (not token.is_stop and token.is_alpha)]
        return query

    def query_to_DNF(self, query: str) -> str:
        query = self.tokenize_query(query)

        for builder in self.query_builders:
            query = builder.build(query)

        query = sympify(query)
        query = to_dnf(query, True)
        return str(query)

    def query(self, query: str, cant: int) -> List[Document]:
        query = self.query_to_DNF(query)
        clauses = query.split(" | ")
        matching_docs = []
        # duda
        for i, doc in self.documents:
            for clause in clauses:
                if clause[0] == "(":
                    clause = clause[1:-1]

                clause_matched = True

                for word in clause.split(" & "):
                    if word[0] == "!":
                        if word[1:] in doc:
                            clause_matched = False
                            break
                    else:
                        if word not in doc:
                            clause_matched = False
                            break
                if clause_matched:
                    matching_docs.append(self.documents[i])
                    break
        return matching_docs
