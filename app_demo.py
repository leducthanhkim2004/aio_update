import streamlit as st
from TextSearch import textSearch

# Initialize textSearch object with paths to your CLIP model and keyframe JSON file
link_bin_v2_file = r"/mnt/f/Luan/AIHCM/Workspace/VN_Multi_User_Video_Search/dataset_extraction/faiss_clipv2_cosine.bin"
link_bin_file = r"/mnt/f/Luan/AIHCM/Workspace/VN_Multi_User_Video_Search/dataset_extraction/faiss_clip.bin"
json_file = r"/mnt/f/Luan/AIHCM/dict/id2img_fps.json"
search_engine = textSearch(link_bin_v2_file, link_bin_file, json_file)

st.title('Text Search Web App')

# Text input for queries
clipQuery = st.text_input('Type your clip query')
clipv2Query = st.text_input('Type your clipv2 query')
ocrQuery = st.text_input('Type your ocr query')

if st.button('Search'):
    cols = st.columns(2) 

    if clipQuery:
        scores, idx_image, link_paths = search_engine.text_search(clipQuery, k=10, model_type='clip')
        data = zip(scores, idx_image, link_paths)
        for idx, img in enumerate(data):
            col = cols[idx % 2]
            with col:
                st.image(img[2], caption=f'Scores: {"%.2f" % img[0]}, Frame index: {img[1]}', use_column_width=True)
    else:
        if ocrQuery:
            scores, idx_image, link_paths = search_engine.ocr_search(ocrQuery, k=10)
            data = zip(scores, idx_image, link_paths)
            for idx, img in enumerate(data):
                col = cols[idx % 2]
                with col:
                    st.image(img[2], caption=f'Scores: {"%.2f" % img[0]}, Frame index: {img[1]}', use_column_width=True)
        else:
            if clipv2Query:
                scores, idx_image, link_paths = search_engine.text_search(clipv2Query, k=10, model_type='clipV2')
                data = zip(scores, idx_image, link_paths)
                # st.write('Scores:', scores)
                # st.write('Image Indexes:', idx_image)
                # st.write('Image Paths:', link_paths)
                for idx, img in enumerate(data):
                    col = cols[idx % 2]
                    with col:
                        st.image(img[2], caption=f'Scores: {"%.2f" % img[0]}, Frame index: {img[1]}', use_column_width=True)
            else:
                st.error('Please enter a query.')
