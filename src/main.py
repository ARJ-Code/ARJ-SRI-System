from sri.models.vectorial import Vectorial
from sri.movie.movie_corpus import MovieCorpus
from sri.models.boolean import Boolean
from sri.models.lsi import LSI
import sys
import nltk
from sri.sri import SRISystem


def main() -> None:

    cant_lines = -1

    try:
        cant_lines = int(sys.argv[1])
    except:
        pass

    corpus = MovieCorpus()
    vectorial_model = Vectorial()
    lsi_model = LSI()
    boolean_model = Boolean()

    sri = SRISystem(corpus, [vectorial_model, lsi_model, boolean_model])
    sri.build(cant_lines)

    nltk.download('wordnet')


if __name__ == "__main__":
    main()
