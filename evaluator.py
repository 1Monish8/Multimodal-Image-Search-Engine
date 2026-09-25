import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Union

from src.config import TEST_QUERIES_PATH, EVAL_REPORT_PATH, EVAL_PLOT_PATH, FAILURE_ANALYSIS_PATH, EVAL_DIR
from src.indexer import FAISSVectorIndexer

class SearchEvaluator:
    """
    Evaluates multimodal search retrieval quality and latency.
    Computes Precision@K, MRR, MAP, and latency statistics across ground-truth test queries.
    """
    def __init__(self, indexer: FAISSVectorIndexer):
        self.indexer = indexer

    def evaluate_query(self, query: Dict, top_k: int = 10) -> Dict:
        """Evaluates a single query against ground truth target relevant IDs."""
        q_text = query["query_text"]
        relevant_ids = set(query["relevant_ids"])

        # Execute text-to-image search
        results, latency_ms = self.indexer.search_by_text(q_text, top_k=top_k)
        retrieved_ids = [r["id"] for r in results]

        # Compute Precision@K for K in [1, 3, 5, 8, 10]
        precision_at_k = {}
        for k in [1, 3, 5, 8, 10]:
            top_k_ids = retrieved_ids[:k]
            hits = len(set(top_k_ids).intersection(relevant_ids))
            precision_at_k[f"P@{k}"] = hits / k if k > 0 else 0.0

        # Compute Reciprocal Rank (RR)
        first_relevant_rank = None
        for rank, r_id in enumerate(retrieved_ids, 1):
            if r_id in relevant_ids:
                first_relevant_rank = rank
                break
        reciprocal_rank = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

        # Compute Average Precision (AP)
        hits_count = 0
        sum_precisions = 0.0
        for rank, r_id in enumerate(retrieved_ids, 1):
            if r_id in relevant_ids:
                hits_count += 1
                sum_precisions += hits_count / rank

        denom = min(len(relevant_ids), top_k) if relevant_ids else 1
        average_precision = sum_precisions / denom if denom > 0 else 0.0

        return {
            "query_id": query["query_id"],
            "query_text": q_text,
            "target_category": query.get("target_category", ""),
            "target_color": query.get("target_color", ""),
            "precision_at_k": precision_at_k,
            "reciprocal_rank": reciprocal_rank,
            "average_precision": average_precision,
            "first_relevant_rank": first_relevant_rank,
            "latency_ms": latency_ms,
            "top_retrieved_items": [
                {
                    "rank": r["rank"],
                    "id": r["id"],
                    "caption": r["caption"],
                    "category": r["category"],
                    "color": r["color"],
                    "similarity_score": r["similarity_score"],
                    "is_relevant": r["id"] in relevant_ids
                }
                for r in results
            ],
            "failure_note": query.get("failure_note", "")
        }

    def run_full_evaluation(self, test_queries_path: Union[str, Path] = TEST_QUERIES_PATH) -> Dict:
        """
        Runs evaluation over all test queries, calculates macro-averaged metrics,
        saves evaluation report, and extracts qualitative failure cases.
        """
        with open(test_queries_path, "r") as f:
            queries = json.load(f)

        print(f"Running evaluation benchmark over {len(queries)} test queries...")

        eval_results = []
        latencies = []

        for q in queries:
            res = self.evaluate_query(q, top_k=10)
            eval_results.append(res)
            latencies.append(res["latency_ms"])

        # Compute macro averages
        p_at_k_means = {}
        for k in [1, 3, 5, 8, 10]:
            p_at_k_means[f"P@{k}"] = float(np.mean([r["precision_at_k"][f"P@{k}"] for r in eval_results]))

        mrr = float(np.mean([r["reciprocal_rank"] for r in eval_results]))
        map_score = float(np.mean([r["average_precision"] for r in eval_results]))

        latency_stats = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99))
        }

        # Identify qualitative failure cases (e.g. Precision@5 < 0.6)
        failure_cases = [
            {
                "query_id": r["query_id"],
                "query_text": r["query_text"],
                "target_category": r["target_category"],
                "precision_at_5": r["precision_at_k"]["P@5"],
                "first_relevant_rank": r["first_relevant_rank"],
                "failure_explanation": r["failure_note"],
                "top_retrieved": r["top_retrieved_items"][:5]
            }
            for r in eval_results if r["precision_at_k"]["P@5"] < 0.6
        ]

        # If few failures meet threshold, include top 3 lowest P@5 queries
        if len(failure_cases) < 3:
            sorted_by_p5 = sorted(eval_results, key=lambda x: x["precision_at_k"]["P@5"])
            failure_cases = [
                {
                    "query_id": r["query_id"],
                    "query_text": r["query_text"],
                    "target_category": r["target_category"],
                    "precision_at_5": r["precision_at_k"]["P@5"],
                    "first_relevant_rank": r["first_relevant_rank"],
                    "failure_explanation": r["failure_note"],
                    "top_retrieved": r["top_retrieved_items"][:5]
                }
                for r in sorted_by_p5[:3]
            ]

        report = {
            "total_queries_evaluated": len(queries),
            "metrics": {
                "precision_at_k": p_at_k_means,
                "mean_reciprocal_rank_MRR": mrr,
                "mean_average_precision_MAP": map_score,
                "latency_stats_ms": latency_stats
            },
            "failure_cases": failure_cases,
            "detailed_query_results": eval_results
        }

        # Save outputs
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(EVAL_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        with open(FAILURE_ANALYSIS_PATH, "w") as f:
            json.dump(failure_cases, f, indent=2)

        self._generate_plots(p_at_k_means, latencies)

        print("\nEvaluation Benchmark Complete!")
        print(f" -> Precision@5: {p_at_k_means['P@5']:.4f}")
        print(f" -> Precision@10: {p_at_k_means['P@10']:.4f}")
        print(f" -> MRR: {mrr:.4f} | MAP: {map_score:.4f}")
        print(f" -> Mean Latency: {latency_stats['mean_ms']:.2f} ms (P95: {latency_stats['p95_ms']:.2f} ms)")
        print(f"Saved evaluation report to {EVAL_REPORT_PATH}")
        print(f"Saved Precision@K plot to {EVAL_PLOT_PATH}")

        return report

    def _generate_plots(self, p_at_k_means: Dict, latencies: List[float]):
        """Generates Precision@K bar plot and latency distribution chart."""
        plt.style.use("ggplot")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Precision@K plot
        ks = list(p_at_k_means.keys())
        values = list(p_at_k_means.values())
        bars = axes[0].bar(ks, values, color="#4C72B0", edgecolor="black", alpha=0.85)
        axes[0].set_title("Multimodal Retrieval Precision @ K", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Metrics", fontsize=11)
        axes[0].set_ylabel("Precision Score", fontsize=11)
        axes[0].set_ylim(0, 1.05)
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Latency histogram plot
        axes[1].hist(latencies, bins=15, color="#55A868", edgecolor="black", alpha=0.85)
        axes[1].set_title("Query Latency Distribution (ms)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Latency (milliseconds)", fontsize=11)
        axes[1].set_ylabel("Query Count", fontsize=11)
        mean_lat = np.mean(latencies)
        axes[1].axvline(mean_lat, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_lat:.2f} ms")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(EVAL_PLOT_PATH, dpi=300)
        plt.close()
