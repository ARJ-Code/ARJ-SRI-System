from ..core import Model, QueryBuilder, Document
from typing import List, Tuple
import json
import numpy as np
from gensim.matutils import corpus2dense
from sympy import sympify, to_dnf
import spacy

nlp = spacy.load('en_core_web_sm')

class Boolean(Model):
    """
    A class for building and manipulating boolean models.

    Args:
        query_builders (List[QueryBuilder]): List of query builders.

    Attributes:
        query_builders (List[QueryBuilder]): List of query builders.
        boolean_data_build (dict): Dictionary containing tokenized documents.

    Methods:
        build_model(tokenized_docs: List[Tuple[str, List[str]]]) -> None:
            Builds the boolean model from tokenized documents.
        _load() -> None:
            Loads the boolean data from a JSON file.
        tokenize_query(query: str) -> List[str]:
            Tokenizes a query and returns a list of relevant terms.
        query_to_DNF(query: str) -> str:
            Converts a query to Disjunctive Normal Form (DNF).

    """
    def __init__(self, query_builders: List[QueryBuilder] = []) -> None:
        super().__init__()
        self.query_builders: List[QueryBuilder] = query_builders

    def build_model(self, tokenized_docs: List[Tuple[str, List[str]]]) -> None:
        """
        Builds the boolean model from tokenized documents.

        Args:
            tokenized_docs (List[Tuple[str, List[str]]]): List of tokenized documents.

        Returns:
            None
        """
        tokenized_docs = [(t, Model._lemma(doc)) for t, doc in tokenized_docs]
        boolean_data_build = {}
        for i in range(len(tokenized_docs)):
            boolean_data_build[tokenized_docs[i][0]
                               ] = tokenized_docs[i][1], tokenized_docs[i][2]

        f = open('data/boolean_data_build.json', 'w')
        json.dump(boolean_data_build, f)
        f.close()

    def _load(self) -> None:
        """
        Loads the boolean data from a JSON file.

        Returns:
            None
        """
        f = open('data/boolean_data_build.json')
        self.boolean_data_build = json.load(f)
        self.boolean_data_build.update(
            {k: (n, set(v)) for k, (n, v) in self.boolean_data_build.items()})
        f.close()

    def tokenize_query(self, query: str) -> List[str]:
        """
        Tokenizes a query and returns a list of relevant terms.

        Args:
            query (str): Input query.

        Returns:
            List[str]: List of tokenized terms.
        """
        exceptions = ["and", "or", "not", "(", ")", "&", "|", "~"]
        query = [token.lemma_ for token in nlp(query.lower()) if token.lemma_ in exceptions or (not token.is_stop and token.is_alpha)]
        return query

    def query_to_DNF(self, query: str) -> str:
        """
        Converts a query to Disjunctive Normal Form (DNF).

        Args:
            query (str): Input query.

        Returns:
            str: Query in DNF.
        """
        query = self.tokenize_query(query)
        for builder in self.query_builders:
            query = builder.build(query)
        query = sympify(query)
        query = to_dnf(query, True)
        return str(query)

    def query(self, query: str, _: int) -> List[Tuple[str, str, float]]:
        """
        Executes a boolean query and returns a list of matching documents.

        Args:
            query (str): The input query in Disjunctive Normal Form (DNF).
            cant (int): The maximum number of matching documents to return.

        Returns:
            List[Document]: A list of matching documents.
        """
        query = self.query_to_DNF(query)
        clauses = query.split(" | ")
        matching_docs = []
        for k, (n, v) in self.boolean_data_build.items():
            for clause in clauses:
                if clause[0] == "(":
                    clause = clause[1:-1]

                clause_matched = True
                for word in clause.split(" & "):
                    if word[0] == "~":
                        if word[1:] in v:
                            clause_matched = False
                            break
                    else:
                        if word not in v:
                            clause_matched = False
                            break
                if clause_matched:
                    matching_docs.append((k, n, 1))
                    break
        return matching_docs
