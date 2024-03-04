from sri.vectorial import Vectorial
from sri.corpus import MovieCorpus
from sri.boolean import Boolean
import sys


def main() -> None:

    cant_lines = -1

    try:
        cant_lines = int(sys.argv[1])
    except:
        pass

    corpus = MovieCorpus()
    vectorial_model = Vectorial()
    boolean_model = Boolean()

    corpus.load(cant_lines)
    vectorial_model.build(corpus.documents)
    # boolean_model.build(corpus.documents)


if __name__ == "__main__":
    main()
