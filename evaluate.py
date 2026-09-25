import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.indexer import FAISSVectorIndexer
from src.evaluator import SearchEvaluator

def main():
    print("==================================================")
    print(" Multimodal Search Engine: Retrieval Evaluation")
    print("==================================================")

    indexer = FAISSVectorIndexer()
    loaded = indexer.load_index()
    
    if not loaded:
        print("Error: Index not found! Please run 'python build_index.py' first.")
        sys.exit(1)

    evaluator = SearchEvaluator(indexer)
    report = evaluator.run_full_evaluation()

    print("\nQuantitative Evaluation Summary:")
    metrics = report["metrics"]
    print(f" -> Precision@1:  {metrics['precision_at_k']['P@1']:.4f}")
    print(f" -> Precision@5:  {metrics['precision_at_k']['P@5']:.4f}")
    print(f" -> Precision@10: {metrics['precision_at_k']['P@10']:.4f}")
    print(f" -> Mean Reciprocal Rank (MRR): {metrics['mean_reciprocal_rank_MRR']:.4f}")
    print(f" -> Mean Average Precision (MAP): {metrics['mean_average_precision_MAP']:.4f}")
    print(f" -> Latency (Mean): {metrics['latency_stats_ms']['mean_ms']:.2f} ms | P95: {metrics['latency_stats_ms']['p95_ms']:.2f} ms")

if __name__ == "__main__":
    main()
