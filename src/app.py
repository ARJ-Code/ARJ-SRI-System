import streamlit as st
from streamlit_searchbox import st_searchbox

st.markdown("<h1 style='text-align: center; color: red;'>Moogle ++</h1>",
            unsafe_allow_html=True)


def search_filter(search_term: str):
    return ['aa', 'aa']


query = st_searchbox(
    search_filter,
)

search = st.button("Search")

if search:
    st.write('aa')
