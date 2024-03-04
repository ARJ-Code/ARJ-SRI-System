from PyDictionary import PyDictionary
from typing import List
from .core import QueryBuilder
from fuzzywuzzy import process


class SpellingChecker(QueryBuilder):
    def build(self, tokens: List[str], words: List[str]):
        result = tokens.copy()

        q = set(tokens)

        for token in tokens:
            w = process.extractOne(token, words, score_cutoff=70)

            if w and w[0] not in q:
                result.append(w[0])

        return result


class Synonymous:
    def build(self, tokens: List[str], words: List[str]):
        dictionary = PyDictionary()
        result = tokens.copy()

        q1 = set(tokens)
        q2 = set(words)

        for token in tokens:
            synonymous = dictionary.synonym(token)
            if synonymous:
                for syn in synonymous:
                    if syn not in q1 and syn in q2:
                        result.append(syn)

        return result
