import streamlit as st
from streamlit_searchbox import st_searchbox
from sri.sri import SRISystem
from sri.vectorial import Vectorial
from sri.corpus import MovieCorpus



corpus = MovieCorpus()
vectorial_model = Vectorial()

sri = SRISystem(corpus, vectorial_model)
sri.load()

st.markdown("<h1 style='text-align: center; color: red;'>Moogle ++</h1>",
            unsafe_allow_html=True)


def search_filter(search_term: str):
    to_search = search_term.split(' ')

    if len(to_search) == 0:
        return []
    
    auto = sri.auto_complete(to_search[-1])

    result = []

    for s in auto:
        to_search[-1] = s
        result.append(' '.join(to_search))

    return result


query = st_searchbox(
    search_filter,
)

search = st.button("Search")

if search:
    st.write('aa')
