from .synonimous import SynonimousDictionary
from typing import List
from .core import QueryBuilder
import spacy
from fuzzywuzzy import process

nlp = spacy.load('en_core_web_sm')


class SpellingChecker(QueryBuilder):
    def build(self, tokens: List[str], words: List[str]):
        result = tokens.copy()

        q = set(tokens)

        for token in tokens:
            if token in words:
                continue

            w = process.extractOne(token, words, score_cutoff=70)

            if w and w[0] not in q:
                result.append(w[0])

        return result


class BooleanQueryBuilder(QueryBuilder):
    def build(self, tokens: List[str]):
        processed_query = ""
        operators = ["and", "or", "not", "(", ")", "&", "|", "!"]
        tokens = " ".join(tokens)
        q = nlp(tokens)

        for i, token in q:
            if token.text in operators:
                processed_query += token.text
            else:
                processed_query += token.text
                if i+1 < len(q) and q[i+1].text not in operators:
                    processed_query += " &"
            processed_query += " "

        processed_query = processed_query.replace(
            " and ", " & ").replace(" or ", " | ").replace(" not ", " ! ")
        return processed_query


class Synonymous:
    def build(self, tokens: List[str], words: List[str]):
        dictionary = SynonimousDictionary()
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
