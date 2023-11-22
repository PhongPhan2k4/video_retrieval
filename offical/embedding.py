import numpy as np
import clip
import torch
from PIL import Image
    
# Text Embedding
class TextEmbedding():
  def __init__(self):
    self.model, _ = clip.load("ViT-B/32")

  def __call__(self, text: str) -> np.ndarray:
    text_inputs = clip.tokenize([text])
    with torch.no_grad():
        text_feature = self.model.encode_text(text_inputs)[0]

    return text_feature.detach().numpy()