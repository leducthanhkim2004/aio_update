import streamlit as st
from TextSearch import textSearch

# Initialize textSearch object with paths to your CLIP model and keyframe JSON file
link_bin_file = r"C:\Users\leduc\ai\data\faiss_clipv2_cosine.bin"
json_file = r"C:\Users\leduc\ai\data\id2img_fps.json"
search_engine = textSearch(link_bin_file, json_file)

st.title('Text Search Web App')

# Text input for queries
query = st.text_input('Type your query')

if st.button('Search'):
    if query:
        scores, idx_image, link_paths = search_engine.text_search(query, k=10)
        st.write('Scores:', scores)
        st.write('Image Indexes:', idx_image)
        st.write('Image Paths:', link_paths)
    else:
        st.error('Please enter a query.')
