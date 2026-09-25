import time
import os
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Union

from src.config import (
    FAISS_INDEX_PATH,
    INDEX_METADATA_PATH,
    EMBEDDINGS_NPY_PATH,
    EMBEDDING_DIM,
    METADATA_PATH
)
from src.models import CLIPEmbeddingEngine

class FAISSVectorIndexer:
    """
    Manages building, saving, loading, and querying FAISS vector index
    over CLIP embeddings for multimodal search.
    """
    def __init__(self):
        self.index = None
        self.metadata_df = None
        self.embeddings = None
        self.embedding_engine = CLIPEmbeddingEngine()

    def build_index_from_dataset(self, metadata_path: Union[str, Path] = METADATA_PATH, batch_size: int = 32):
        """
        Loads dataset metadata, generates image CLIP embeddings, and builds FAISS IndexFlatIP.
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please run prepare_dataset.py first.")

        df = pd.read_csv(metadata_path)
        print(f"Loaded dataset metadata with {len(df)} records.")

        image_paths = df["image_path"].tolist()
        print(f"Generating CLIP embeddings for {len(image_paths)} images in batches of {batch_size}...")

        start_time = time.time()
        self.embeddings = self.embedding_engine.encode_images(image_paths, batch_size=batch_size)
        encoding_time = time.time() - start_time
        print(f"Embeddings extracted in {encoding_time:.2f}s. Shape: {self.embeddings.shape}")

        # Initialize FAISS Index (Inner Product for Cosine Similarity on L2-normalized vectors)
        print(f"Initializing FAISS IndexFlatIP (dimension={EMBEDDING_DIM})...")
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(self.embeddings)
        self.metadata_df = df.copy()

        print(f"FAISS index successfully built with {self.index.ntotal} vectors.")
        self.save_index()

    def save_index(self):
        """Saves binary FAISS index, metadata Parquet file, and numpy embeddings to disk."""
        if self.index is None or self.metadata_df is None:
            raise ValueError("No active index to save. Build or load an index first.")

        print(f"Saving binary FAISS index to {FAISS_INDEX_PATH}...")
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

        print(f"Saving metadata to {INDEX_METADATA_PATH}...")
        self.metadata_df.to_parquet(INDEX_METADATA_PATH, index=False)

        print(f"Saving numpy embeddings to {EMBEDDINGS_NPY_PATH}...")
        np.save(EMBEDDINGS_NPY_PATH, self.embeddings)

        print("Index, metadata, and embeddings saved successfully!")

    def load_index(self) -> bool:
        """Loads index and metadata from disk if available."""
        if not (FAISS_INDEX_PATH.exists() and INDEX_METADATA_PATH.exists()):
            return False

        print(f"Loading binary FAISS index from {FAISS_INDEX_PATH}...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        print(f"Loading metadata from {INDEX_METADATA_PATH}...")
        self.metadata_df = pd.read_parquet(INDEX_METADATA_PATH)

        if EMBEDDINGS_NPY_PATH.exists():
            self.embeddings = np.load(EMBEDDINGS_NPY_PATH)

        print(f"FAISS Index loaded with {self.index.ntotal} vectors.")
        return True

    def search_by_vector(self, query_vector: np.ndarray, top_k: int = 8) -> List[Dict]:
        """
        Searches FAISS index using a normalized 1D or 2D query vector (512-D).
        Returns top_k result items with cosine similarity scores and metadata.
        """
        if self.index is None:
            raise ValueError("Index not loaded. Build or load an index first.")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # L2 normalize query vector
        query_vector = self.embedding_engine._l2_normalize(query_vector)

        # Search FAISS index
        similarities, indices = self.index.search(query_vector, top_k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata_df):
                continue
            row = self.metadata_df.iloc[idx].to_dict()
            row["similarity_score"] = float(sim)
            row["rank"] = len(results) + 1
            results.append(row)

        return results

    def search_by_text(self, query_text: str, top_k: int = 8) -> Tuple[List[Dict], float]:
        """
        Text-to-Image Search: Encodes text query, queries FAISS index, returns top matches + latency.
        """
        start_t = time.time()
        text_vector = self.embedding_engine.encode_text(query_text)
        results = self.search_by_vector(text_vector, top_k=top_k)
        latency_ms = (time.time() - start_t) * 1000.0
        return results, latency_ms

    def search_by_image(self, image: Union[Image.Image, str], top_k: int = 8) -> Tuple[List[Dict], float]:
        """
        Image-to-Image / Image-to-Text Search: Encodes input image, queries FAISS index, returns top matches + latency.
        """
        start_t = time.time()
        image_vector = self.embedding_engine.encode_images([image])
        results = self.search_by_vector(image_vector, top_k=top_k)
        latency_ms = (time.time() - start_t) * 1000.0
        return results, latency_ms

    def search_hybrid(self, image: Union[Image.Image, str], text: str, alpha: float = 0.5, top_k: int = 8) -> Tuple[List[Dict], float]:
        """
        Hybrid Multimodal Search: Blends text embedding and image embedding weighted by alpha.
        blend = alpha * text_vec + (1 - alpha) * image_vec
        """
        start_t = time.time()
        text_vector = self.embedding_engine.encode_text(text)
        image_vector = self.embedding_engine.encode_images([image])

        blended_vec = alpha * text_vector + (1.0 - alpha) * image_vector
        blended_vec = self.embedding_engine._l2_normalize(blended_vec)

        results = self.search_by_vector(blended_vec, top_k=top_k)
        latency_ms = (time.time() - start_t) * 1000.0
        return results, latency_ms

    def search_with_negative_prompt(self, positive_text: str, negative_text: str, beta: float = 0.4, top_k: int = 8) -> Tuple[List[Dict], float]:
        """
        Negative Prompting Search via Vector Arithmetic:
        v_final = normalize(v_pos - beta * v_neg)
        Subtracts unwanted attribute features in the 512-D latent space.
        """
        start_t = time.time()
        pos_vec = self.embedding_engine.encode_text(positive_text)
        neg_vec = self.embedding_engine.encode_text(negative_text)

        # Vector arithmetic in embedding space
        blended_vec = pos_vec - beta * neg_vec
        blended_vec = self.embedding_engine._l2_normalize(blended_vec)

        results = self.search_by_vector(blended_vec, top_k=top_k)
        latency_ms = (time.time() - start_t) * 1000.0
        return results, latency_ms

    def add_image_to_index(self, image: Union[Image.Image, str], caption: str, category: str, color: str = "custom", tags: str = "") -> Dict:
        """
        Dynamically ingests a new image into the live FAISS index and persists to disk.
        """
        from src.config import IMAGES_DIR
        
        # Save image file locally
        next_id = len(self.metadata_df) + 1
        safe_name = f"custom_{next_id:04d}_{category}.jpg"
        save_path = IMAGES_DIR / safe_name

        if isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
            import urllib.request
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(image, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp, open(save_path, 'wb') as f:
                f.write(resp.read())
            pil_img = Image.open(save_path).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            pil_img.save(save_path, "JPEG", quality=95)
        else:
            pil_img = Image.open(image).convert("RGB")
            pil_img.save(save_path, "JPEG", quality=95)

        # Extract CLIP embedding
        new_vec = self.embedding_engine.encode_images([pil_img]) # shape (1, 512)

        # Append to live FAISS index
        self.index.add(new_vec)

        # Append to metadata DataFrame
        new_row = {
            "id": next_id,
            "filename": safe_name,
            "caption": caption,
            "category": category,
            "color": color,
            "tags": tags if tags else f"{category}, {color}, custom",
            "image_path": str(save_path)
        }
        self.metadata_df = pd.concat([self.metadata_df, pd.DataFrame([new_row])], ignore_index=True)

        # Append to embeddings numpy array if loaded
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_vec])

        # Persist updated index and metadata to disk
        self.save_index()
        return new_row
