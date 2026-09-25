import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.dataset_loader import create_dataset, create_test_queries
from src.config import METADATA_PATH

def main():
    print("==================================================")
    print(" Multimodal Search Engine: Dataset Preparation")
    print("==================================================")
    
    # Create images and metadata
    df = create_dataset()
    
    # Create evaluation test queries
    queries = create_test_queries(df)
    
    print("\nDataset preparation completed successfully!")
    print(f"Total Images: {len(df)}")
    print(f"Categories: {df['category'].nunique()} ({', '.join(df['category'].unique())})")
    print(f"Test Queries: {len(queries)}")

if __name__ == "__main__":
    main()
