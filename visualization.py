import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple

def project_embeddings_2d(embeddings: np.ndarray, method: str = "pca") -> np.ndarray:
    """
    Projects 512-D CLIP embeddings into 2D space using PCA or t-SNE.
    """
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    else:
        reducer = PCA(n_components=2, random_state=42)
        
    coords_2d = reducer.fit_transform(embeddings)
    return coords_2d


def plot_embedding_space(coords_2d: np.ndarray, metadata_df: pd.DataFrame, title: str = "CLIP Multimodal Embedding Space (2D Projection)") -> plt.Figure:
    """
    Generates a scatter plot of 2D projected dataset embeddings colored by category.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.style.use("ggplot")
    
    categories = metadata_df["category"].unique()
    palette = sns.color_palette("tab10", len(categories))

    for idx, cat in enumerate(categories):
        mask = metadata_df["category"] == cat
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            label=cat,
            color=palette[idx],
            alpha=0.75,
            edgecolors="none",
            s=45
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_query_neighborhood_2d(coords_2d: np.ndarray, metadata_df: pd.DataFrame, query_coord: np.ndarray, result_indices: List[int], query_label: str) -> plt.Figure:
    """
    Plots the overall embedding space, highlighting the position of a user query vector
    and its retrieved top-K nearest neighbors with connecting lines.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.style.use("ggplot")

    # Plot all background dataset points
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], color="#BDC3C7", alpha=0.4, label="Dataset Images", s=30)

    # Plot retrieved top-k items
    retrieved_coords = coords_2d[result_indices]
    ax.scatter(retrieved_coords[:, 0], retrieved_coords[:, 1], color="#2ECC71", alpha=0.9, label="Top Retrieved Neighbors", s=90, edgecolors="black")

    # Plot query point
    ax.scatter(query_coord[0], query_coord[1], color="#E74C3C", marker="*", s=250, label=f"Query: '{query_label}'", edgecolors="black", zorder=5)

    # Draw lines connecting query to top retrieved neighbors
    for coord in retrieved_coords:
        ax.plot([query_coord[0], coord[0]], [query_coord[1], coord[1]], color="#E74C3C", linestyle=":", alpha=0.6, linewidth=1.2)

    ax.set_title(f"Query Vector & Nearest Neighbors in Embedding Space", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig
