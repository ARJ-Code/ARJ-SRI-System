import nltk
from nltk.corpus import wordnet

class SynonimousDictionary:
    def __init__(self) -> None:
        self.download_wordnet()

    def download_wordnet():
        try:
            nltk.data.find('wordnet', 'data/nltk_data/corpora')
        except LookupError:
            nltk.download('wordnet', download_dir='data/nltk_data')

    def synonym(word: str):
        synsets = wordnet.synsets(word)
        synonyms = [lemma.name().lower() for synset in synsets for lemma in synset.lemmas()]

        synonymus_set = set()
        for word in synonyms:
            synonymus_set.add(word)

        return [word for word in synonymus_set]