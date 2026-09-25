import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.indexer import FAISSVectorIndexer
from src.config import METADATA_PATH

def main():
    print("==================================================")
    print(" Multimodal Search Engine: Building FAISS Index")
    print("==================================================")
    
    indexer = FAISSVectorIndexer()
    indexer.build_index_from_dataset(metadata_path=METADATA_PATH, batch_size=32)
    
    print("\nFAISS Indexing pipeline finished successfully!")

if __name__ == "__main__":
    main()
