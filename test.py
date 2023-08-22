
import model
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted
from typing import List, Tuple
import streamlit as st

IMAGE_KEYFRAME_PATH = "Keyframes"
FEATURES_PATH = "clip-features-b1\\clip-features"

st.title("Video Retrieval")

def visualize(imgs: List[Image.Image]) -> None:
    rows = len(imgs) // 3
    if not rows:
        rows += 1
    cols = len(imgs) // rows
    if rows * cols < len(imgs):
        rows += 1
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    st.image(grid)
    
visual_features_df = model.indexing_methods(FEATURES_PATH)
text_embedd = model.TextEmbedding()

def query(text_query = "", topk = 10, measure_method = 'cosine_similarity'):
    # Tạo vector biểu diễn cho câu truy vấn văn bản
    text_feat_arr = text_embedd(text_query)

    # Chuyển DataFrame thành danh sách tuples
    visual_features_db = visual_features_df.to_records(index=False).tolist()

    # Thực hiện tìm kiếm và hiển thị kết quả
    search_result = model.search_engine(text_feat_arr, visual_features_db, topk, measure_method)
    images = model.read_image(search_result)
    visualize(images)
    
topk = st.slider("Number of images to show:", 1, 30)
measure_method = st.selectbox("Measure method:", ("cosine_similarity", "l1_norm"))
text_query = st.text_input("Enter query:")

search = st.button("Search")
if search:
    query(text_query, topk, measure_method)
