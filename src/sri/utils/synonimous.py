from nltk.corpus import wordnet

class SynonimousDictionary:
    def synonym(self, word: str):
        synsets = wordnet.synsets(word)
        synonyms = [lemma.name().lower()
                    for synset in synsets for lemma in synset.lemmas()]

        synonymus_set = set()
        for word in synonyms:
            synonymus_set.add(word)

        return [word for word in synonymus_set]
