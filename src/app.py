import streamlit as st
from streamlit_searchbox import st_searchbox
from sri.sri import SRISystem
from sri.vectorial import Vectorial
from sri.lsi import LSI
from sri.boolean import Boolean
from sri.corpus import MovieCorpus
from sri.query_builder import SpellingChecker, Synonymous
import sys

cant_lines = -1
try:
    cant_lines = int(sys.argv[1])
except:
    pass


corpus = MovieCorpus()

vectorial_model = Vectorial(query_builders=[SpellingChecker(), Synonymous()])
lsi_model = LSI(query_builders=[SpellingChecker(), Synonymous()])
boolean_model = Boolean()

sri = SRISystem(corpus, [vectorial_model, lsi_model])
sri.load(cant_lines)

st.markdown("<h1 style='text-align: center; color: red;'>Moogle ++</h1>",
            unsafe_allow_html=True)

sri_models = {
    "Vectorial": 0,
    "LSI": 1,
}

# Agregar un selectbox para seleccionar el modelo SRI
selected_model_name = st.selectbox(
    'Select the information retrieval system (SRI):',
    options=list(sri_models.keys()),
    index=0
)

# Seleccionar el modelo SRI basado en la elección del usuario
sri.change_selected(sri_models[selected_model_name])


def search_filter(search_term: str):
    to_search = search_term.split(' ')

    if len(to_search) == 0:
        return []

    auto = sri.auto_complete(to_search[-1])

    result = [search_term]

    for s in auto:
        to_search[-1] = s
        result.append(' '.join(to_search))

    return result


query = st_searchbox(
    search_filter,
)

search = st.button("Search")

if search:
    if isinstance(query, str):
        result = sri.query(query)

        for r in result:
            st.write(r.title)

        if len(result) == 0:
            st.write("No results")
