from sri.vectorial import Vectorial
from sri.corpus import MovieCorpus
import sys


def main() -> None:

    cant_lines = -1

    try:
        cant_lines = int(sys.argv[1])
    except:
        pass
    
    corpus = MovieCorpus()
    vectorial_model = Vectorial()

    corpus.load(cant_lines)
    vectorial_model.build(corpus.documents)


if __name__ == "__main__":
    main()
