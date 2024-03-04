import streamlit as st
from streamlit_searchbox import st_searchbox
from sri.sri import SRISystem
from sri.vectorial import Vectorial
from sri.corpus import MovieCorpus
from sri.query_builder import SpellingChecker
import sys

cant_lines = -1
try:
    cant_lines = int(sys.argv[1])
except:
    pass


corpus = MovieCorpus()
vectorial_model = Vectorial(query_builders=[SpellingChecker()])

sri = SRISystem(corpus, vectorial_model)
sri.load(cant_lines)

st.markdown("<h1 style='text-align: center; color: red;'>Moogle ++</h1>",
            unsafe_allow_html=True)


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
