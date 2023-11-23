from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
    
class TextEmbedding():
  def __init__(self):
    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

  def __call__(self, text: str) -> np.ndarray:
    input = self.processor(text, images=Image.fromarray(np.array([0,0,0])), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_feature = self.model(**input)['text_embeds'][0]

    return text_feature.detach().numpy()
