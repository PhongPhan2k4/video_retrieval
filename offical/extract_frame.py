from PIL import Image
import cv2
import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
import tensorflow as tf

model = tf.keras.applications.MobileNetV3Large(weights='imagenet')

def extract_features(frame):
    frame = cv2.resize(frame, (224, 224))  
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    frame = tf.keras.applications.mobilenet_v3.preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)  
    
    # Extract features from the frame
    features = model.predict(frame)
    
    return features

def cosine_similarity(features1, features2):
    # Calculate the cosine similarity between the two feature vectors
    similarity = np.dot(features1, features2.T) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity
    
def extract_frames(video_path, output_path):
    # Mở video
    video = cv2.VideoCapture(video_path)
    _, frame = video.read()
    frame_count = 1
    pre_frame_arr = extract_features(frame)

    while True:
        # Đọc từng frame
        ret, frame = video.read()

        # Kiểm tra xem frame có được đọc thành công không
        if not ret:
            break
        
        frame_arr = extract_features(frame)
        if cosine_similarity(pre_frame_arr, frame_arr) < 0.7:
            # Lưu frame thành ảnh
            cv2.imwrite(output_path + "/" + f"{frame_count}.jpg", frame)
            
        pre_frame_arr = frame_arr

        frame_count += 1

    # Đóng video
    video.release()


# Đường dẫn đến video
videos_path = R"Videos/Videos_L36/video"
for video_name in natsorted(os.listdir(videos_path)):
    video_path = os.path.join(videos_path, video_name)
    output_path = 'extract_frame/L36/' + video_name.split('.')[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        extract_frames(video_path, output_path)

