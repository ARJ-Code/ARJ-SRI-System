from sri.models.vectorial import Vectorial
from sri.movie.movie_corpus import MovieCorpus
from sri.models.boolean import Boolean
from sri.models.lsi import LSI
import sys
import nltk
from sri.sri import SRISystem
from sri.ir_dataset import IRDataset


def main() -> None:

    cant = -1

    try:
        cant = int(sys.argv[1])
    except:
        pass

    corpus = IRDataset("cranfield")
    vectorial_model = Vectorial()
    lsi_model = LSI()
    # boolean_model = Boolean()

    sri = SRISystem([vectorial_model, lsi_model])
    sri.build(corpus, cant)

    nltk.download('wordnet')


if __name__ == "__main__":
    main()
