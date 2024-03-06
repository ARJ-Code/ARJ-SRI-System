import streamlit as st
from streamlit_searchbox import st_searchbox
from sri.sri import SRISystem
from sri.models.vectorial import Vectorial
from sri.models.lsi import LSI
from sri.models.boolean import Boolean
from sri.movie.movie_corpus import MovieCorpus
from sri.query_builder import BooleanQueryBuilder, SpellingChecker, Synonymous
import sys

cant_lines = -1
try:
    cant_lines = int(sys.argv[1])
except:
    pass


corpus = MovieCorpus()

vectorial_model = Vectorial(query_builders=[SpellingChecker(), Synonymous()])
lsi_model = LSI(query_builders=[SpellingChecker(), Synonymous()])
boolean_model = Boolean(query_builders=[BooleanQueryBuilder()])

st.session_state.sri = SRISystem(corpus, [vectorial_model, lsi_model, boolean_model])
sri = st.session_state.sri
sri.load(cant_lines)

st.markdown("<h1 style='text-align: center; color: red;'>Moogle ++</h1>",
            unsafe_allow_html=True)

sri_models = {
    "Vectorial": 0,
    "LSI": 1,
    "boolean": 2,
}

selected_model_name = st.selectbox(
    'Select the information retrieval system (SRI):',
    options=list(sri_models.keys()),
    index=0
)

sri.change_selected(sri_models[selected_model_name])


def search_filter(search_term: str):
    """
    Filters and modifies a search term for auto-completion.

    Args:
        search_term (str): The input search term.

    Returns:
        List[str]: A list of modified search terms.
    """
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

search = st.button("🔍 Search")

if 'result' not in st.session_state:
    st.session_state.result = []

if search:
    if isinstance(query, str):
        st.session_state.result = sri.query(query)
        if len(st.session_state.result) == 0:
            st.write("No results")

st.markdown("""
            <style>
                div[data-testid="column"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="column"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)

for i, r in enumerate(st.session_state.result):
    # Display the title of the result
    st.write(r.title)

    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    if col1.button("✔", key=f"{i},1"):
        sri.add_relevant(r.title)

    if col2.button("✘", key=f"{i},2"):
        sri.add_non_relevant(r.title)
