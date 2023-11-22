import numpy as np
import os
from PIL import Image
from natsort import natsorted
import pandas as pd
import embedding
from tqdm import tqdm

image_embedder = embedding.ImageEmbedding()

def extract_features(folder_path):
    dict = {'paths_frame':[], 'features':[]}
    videos = natsorted([file for file in os.listdir(folder_path)])
    for video in tqdm(videos):
        video_path = os.path.join(folder_path, video)
        for image in tqdm(natsorted([file for file in os.listdir(video_path)])):
            frame_path = os.path.join(video_path, image)
            dict['paths_frame'].append(frame_path)
            img = Image.open(frame_path)
            dict['features'].append(image_embedder(img))

    return pd.DataFrame(dict)


df = extract_features('extract_frame/L22')
df.to_csv('features/L22.csv', index=False)