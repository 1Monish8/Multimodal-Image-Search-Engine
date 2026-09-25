import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import CLIPProcessor, CLIPModel
from src.config import MODEL_NAME, DEVICE, EMBEDDING_DIM

class CLIPEmbeddingEngine:
    """
    Singleton class for loading and running CLIP multimodal embedding model.
    Handles image encoding, text encoding, and L2 vector normalization.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CLIPEmbeddingEngine, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        print(f"Loading CLIP model '{MODEL_NAME}' onto device: '{DEVICE}'...")
        self.device = DEVICE
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model.eval()
        print("CLIP model initialized successfully!")

    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Applies L2 normalization to force unit magnitude (norm = 1.0)."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return (vectors / norms).astype(np.float32)

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encodes text query string(s) into 512-D normalized CLIP text vectors.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            
            # Extract raw tensor to CPU numpy
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            features_np = text_features.cpu().numpy()
            all_embeddings.append(features_np)

        concatenated = np.vstack(all_embeddings)
        return self._l2_normalize(concatenated)

    @torch.no_grad()
    def encode_images(self, images: List[Union[Image.Image, str]], batch_size: int = 32) -> np.ndarray:
        """
        Encodes PIL Image objects or image file paths into 512-D normalized CLIP image vectors.
        """
        pil_images = []
        for img in images:
            if isinstance(img, (str, torch.Tensor)):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        all_embeddings = []
        for i in range(0, len(pil_images), batch_size):
            batch_imgs = pil_images[i:i + batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            features_np = image_features.cpu().numpy()
            all_embeddings.append(features_np)

        concatenated = np.vstack(all_embeddings)
        return self._l2_normalize(concatenated)

    def compute_similarity(self, text: str, image: Union[Image.Image, str]) -> float:
        """
        Computes cosine similarity (dot product of L2 normalized vectors) between a text and image.
        """
        text_vec = self.encode_text(text) # shape (1, 512)
        img_vec = self.encode_images([image]) # shape (1, 512)
        similarity = float(np.dot(text_vec[0], img_vec[0]))
        return similarity
