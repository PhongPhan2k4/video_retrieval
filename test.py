

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



import os
import numpy as np

from tqdm import tqdm
from PIL import Image

import torch
import clip

IMAGE_KEYFRAME_PATH = "Keyframes"
FEATURES_PATH = "clip-features-b1\\clip-features"

class TextEmbedding():
  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model, _ = clip.load("ViT-B/16", device=self.device)

  def __call__(self, text: str) -> np.ndarray:
    text_inputs = clip.tokenize([text]).to(self.device)
    with torch.no_grad():
        text_feature = self.encode_text(text_inputs)[0]

    return text_feature.detach().cpu().numpy()

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from typing import List, Tuple

def indexing_methods(features_path: str) -> pd.DataFrame:
    data = {'video_name': [], 'frame_index': [], 'features_vector': []}

    npy_files = natsorted([file for file in os.listdir(features_path) if file.endswith(".npy")])

    for feat_npy in tqdm(npy_files):
        video_name = feat_npy.split('.')[0]
        feats_arr = np.load(os.path.join(features_path, feat_npy))

        # Lặp qua từng dòng trong feats_arr, mỗi dòng là một frame
        for idx, feat in enumerate(feats_arr):
            data['video_name'].append(video_name)
            data['frame_index'].append(idx)
            data['features_vector'].append(feat)

    df = pd.DataFrame(data)
    return df

FEATURES_PATH = "clip-features-b1\\clip-features"
visual_features_df = indexing_methods(FEATURES_PATH)

visual_features_df.head()

def search_engine(query_arr: np.array,
                  db: list,
                  topk:int=10,
                  measure_method: str="cosine_similarity") -> List[dict]:

    '''Duyệt tuyến tính và tính độ tương đồng giữa 2 vector'''
    measure = []
    for ins_id, instance in enumerate(db):
        video_name, idx, feat_vec = instance

        distance = 0
        if measure_method == "cosine_similarity":
            dot_product = query_arr @ feat_vec
            query_norm = np.linalg.norm(query_arr)
            feat_norm = np.linalg.norm(feat_vec)
            cosine_similarity = dot_product / (query_norm * feat_norm)
            distance = 1 - cosine_similarity
        else:
            distance = np.linalg.norm(query_arr - feat_vec, ord=1)

        measure.append((ins_id, distance))

    '''Sắp xếp kết quả'''
    measure = sorted(measure, key=lambda x:x[1])

    '''Trả về top K kết quả'''
    search_result = []
    for instance in measure[:topk]:
        ins_id, distance = instance
        video_name, idx = db[ins_id][0], db[ins_id][1]

        search_result.append({"video_name": video_name,
                              "keyframe_id": idx+1,
                              "score": distance})

    # Đảm bảo trả về đúng topk kết quả
    while len(search_result) < topk and len(measure) > len(search_result):
        ins_id, distance = measure[len(search_result)]
        video_name, idx = db[ins_id][0], db[ins_id][1]
        search_result.append({"video_name": video_name,
                              "keyframe_id": idx,
                              "score": distance})

    return search_result


import os
from typing import List
from PIL import Image

def read_image(results: List[dict]) -> List[Image.Image]:
    images = []

    for res in results:
        video_name = res["video_name"]
        keyframe_id = res["keyframe_id"]
        video_folder = os.path.join(IMAGE_KEYFRAME_PATH, "Keyframes_" + video_name[:3], video_name)

        if os.path.exists(video_folder):
            image_files = sorted(os.listdir(video_folder))

            if keyframe_id < len(image_files):
                image_file = image_files[keyframe_id]
                image_path = os.path.join(video_folder, image_file)
                image = Image.open(image_path)
                images.append(image)
            else:
                print(f"Keyframe id {keyframe_id} is out of range for video {video_name}.")

    return images


st.title("Video Retrieval by Phong Phan")

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
    
visual_features_df = indexing_methods(FEATURES_PATH)
text_embedd = TextEmbedding()

def query(text_query = "", topk = 10, measure_method = 'cosine_similarity'):
    # Tạo vector biểu diễn cho câu truy vấn văn bản
    text_feat_arr = text_embedd(text_query)

    # Chuyển DataFrame thành danh sách tuples
    visual_features_db = visual_features_df.to_records(index=False).tolist()

    # Thực hiện tìm kiếm và hiển thị kết quả
    search_result = search_engine(text_feat_arr, visual_features_db, topk, measure_method)
    images = read_image(search_result)
    visualize(images)
    
topk = st.slider("Number of images to show:", 1, 30)
measure_method = st.selectbox("Measure method:", ("cosine_similarity", "l1_norm"))
text_query = st.text_input("Enter query:")

search = st.button("Search")
if search:
    query(text_query, topk, measure_method)
