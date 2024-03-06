from nltk.corpus import wordnet

class SynonimousDictionary:
    def synonym(self, word: str):
        """
        Finds synonyms for a given word using WordNet.

        Args:
            word (str): The input word.

        Returns:
            list: A list of unique synonyms for the input word.
        """
        synsets = wordnet.synsets(word)
        synonyms = [lemma.name().lower()
                    for synset in synsets for lemma in synset.lemmas()]

        synonymus_set = set()
        for word in synonyms:
            synonymus_set.add(word)

        return [word for word in synonymus_set]
