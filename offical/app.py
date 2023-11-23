import streamlit as st
from PIL import Image
import faiss
import numpy as np
from modules.embedding import TextEmbedding
import pandas as pd
import requests
import json
from googletrans import Translator

st.set_page_config(layout="wide")
st.title('W1 Doppelganger')

@st.cache_resource
def load():
    text_embedder = TextEmbedding()
    btc = pd.read_csv('indexing.csv')
    extract = pd.read_csv('extract_indexing.csv')
    index = faiss.read_index('faiss_index.faiss')
    translator = Translator()
    return text_embedder, btc, extract, index, translator
text_embedder, btc, extract, index, translator = load()

@st.cache_data
def get_session(username, password):
    url = "https://eventretrieval.one/api/v1/login"
    
    data = {"username" : username,
            "password" : password
            }

    response = requests.post(url, data=json.dumps(data))
    print(response.json()["sessionId"])
    if response.status_code == 200:
        return response.json()["sessionId"]
    else:
        return None
    
@st.cache_data
def search_fn(query_text):
    query_text = translator.translate(query_text, dest='en').text
    st.write(query_text)
    query_arr = [text_embedder(query_text)]
    embedding_query = np.array(query_arr)
    _,I = index.search(embedding_query, num_search)
    return btc.iloc[I[0]].reset_index()
    
@st.cache_data
def submit(video, frame, session):
    url = "https://eventretrieval.one/api/v1/submit"
    
    params = {"item": video,
              "frame": frame,
              "session": session
              }

    # Thực hiện GET request
    response = requests.get(url, params=params)

    return response.json()

# session = get_session("doppelganger", "uu1AShei")
# st.sidebar.subheader("Reset Session Id")
# get = st.sidebar.button("Reset")
# if get:
#     session = get_session("doppelganger", "uu1AShei")

session = st.sidebar.text_input('session id','node010iki7qadx0gb1fphcw39i1v852802')

st.sidebar.subheader("Submit")
input = st.sidebar.text_input("Enter")
sub = st.sidebar.button("Submit")
if sub:
    video, frame = input.split(',')
    st.sidebar.json(submit(video, frame, session))
    
type = st.sidebar.selectbox('Search by', ['Query', 'Video'])
st.text_area('Original query')
if type == 'Query':
    num_search = st.sidebar.selectbox('Number of images to search', [100, 160, 200])
    # language = st.sidebar.selectbox('Language', ['Vietnamese', 'English'])
    query_text = st.text_input('Enter a text query:')
    search = st.button('Search')

    # select box
    images_per_row = 4
    num_rows = num_search // 4

    if search and query_text != '':
        df = search_fn(query_text)
        # display
        for i in range(num_rows):
            cols = st.columns(images_per_row)
            for j in range(images_per_row):
                idx = i*images_per_row + j
                img = Image.open(df['paths_frame'][idx]).convert('RGB')
                with cols[j]:
                    st.image(img, use_column_width=True)
                    st.code(f"{df['videos_name'][idx]},{df['frames_id'][idx]}")

   
else:
    keyframe = st.selectbox('Keyframe', ['Extract', 'BTC'])
    input = st.text_input('enter video, frame')
    if input != '':
        video, frame = input.split(',')
        if keyframe == 'BTC':
            index = btc.query(f"videos_name == '{video}' & frames_id == {frame}").index[0]
            df = btc.loc[range(index - 10, index + 10)].reset_index()
            num_image = 20
        else:
            df = extract.query(f"videos_name == '{video}' & frames_id >= {int(frame) - 1000} & frames_id <= {int(frame) + 1000}")
            df = df.reset_index()
            num_image = df.shape[0]
        num_rows = 1 + num_image // 4
    search = st.button('Search')
    if search:
        for i in range(num_rows):
            cols = st.columns(4)
            for j in range(4):
                id = i * 4 + j
                if id < num_image:
                    frame_id = df['frames_id'][id]
                    img = Image.open(df['paths_frame'][id]).convert('RGB')
                    with cols[j]:
                        st.image(img, use_column_width=True)
                        st.code(f'{video},{frame_id}')


    
