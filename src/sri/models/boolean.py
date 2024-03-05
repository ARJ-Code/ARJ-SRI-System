from ..core import Model, QueryBuilder, Document
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

    def build_model(self, tokenized_docs: List[Tuple[str, List[str]]]):
        tokenized_docs = [(t, Model._lemma(doc)) for t, doc in tokenized_docs]

        boolean_data_build = {}

        for i in range(len(tokenized_docs)):
            boolean_data_build[tokenized_docs[i][0]] = tokenized_docs[i][1]

        f = open('data/boolean_data_build.json', 'w')
        json.dump(boolean_data_build, f)
        f.close()

    def _load(self):
        f = open('data/boolean_data_build.json')
        self.boolean_data_build = json.load(f)
        f.close()

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

        for i in range(len(self.documents)):
            for clause in clauses:
                if clause[0] == "(":
                    clause = clause[1:-1]

                clause_matched = True

                for word in clause.split(" & "):
                    if word[0] == "!":
                        if word[1:] in self.boolean_data_build[self.documents[i].title]:
                            clause_matched = False
                            break
                    else:
                        if word not in self.boolean_data_build[self.documents[i].title]:
                            clause_matched = False
                            break
                if clause_matched:
                    matching_docs.append(self.documents[i])
                    break
        return matching_docs
