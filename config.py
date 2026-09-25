import os
from pathlib import Path
import torch

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset Paths
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
METADATA_PATH = DATASET_DIR / "metadata.csv"
TEST_QUERIES_PATH = DATASET_DIR / "test_queries.json"

# Index & Storage Paths
INDEX_DIR = BASE_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
INDEX_METADATA_PATH = INDEX_DIR / "metadata.parquet"
EMBEDDINGS_NPY_PATH = INDEX_DIR / "embeddings.npy"

# Evaluation Output Paths
EVAL_DIR = BASE_DIR / "eval_results"
EVAL_REPORT_PATH = EVAL_DIR / "evaluation_report.json"
EVAL_PLOT_PATH = EVAL_DIR / "precision_at_k.png"
FAILURE_ANALYSIS_PATH = EVAL_DIR / "failure_analysis.json"

# Model Configs
MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512

# Device Auto-detection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Ensure directories exist
for d in [DATASET_DIR, IMAGES_DIR, INDEX_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)
