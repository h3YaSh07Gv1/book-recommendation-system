import sys
import os
import pandas as pd
import numpy as np
import time
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import data_loader
from src.engine.recommender import Recommender

def benchmark_model(model_name, sample_size=50, top_k=5):
    """
    Benchmarks a specific embedding model on the recommender system.
    Returns a dict of metrics.
    """
    print(f"\n--- Benchmarking {model_name} ---")

    # Load data
    documents = data_loader.load_documents("data/tagged_description.txt")
    books = data_loader.load_books("data/books_with_redirect_links(emotions).csv")
    books["isbn13"] = books["isbn13"].astype(str)

    # Create recommender with specific model
    # We need to modify Recommender to accept embedding_model_name
    start_time = time.time()
    try:
        recommender = Recommender(documents, embedding_model_name=model_name)
        load_time = time.time() - start_time
        print(f"Model load time: {load_time:.2f}s")
    except Exception as e:
        print(f"Failed to initialize model {model_name}: {e}")
        return {'model': model_name, 'error': str(e)}

    # Filter books with valid categories
    valid_books = books[books["simple_category"].notna() & (books["simple_category"] != "Unknown")]

    # Sample books
    sample = valid_books.sample(min(sample_size, len(valid_books)))

    print(f"Evaluating on {len(sample)} books...")

    y_true = []
    y_pred = []

    correct_retrievals = 0
    total_retrievals = 0

    mrr_sum = 0
    hit_rate_sum = 0
    ndcg_sum = 0

    inference_times = []

    for _, row in sample.iterrows():
        # Construct query
        query = f"{row['title']}. {row['description'][:100]}"
        ground_truth_category = row["simple_category"]

        # Time the search
        start_inf = time.time()
        recs = recommender.hybrid_search(query, k=top_k)
        inf_time = time.time() - start_inf
        inference_times.append(inf_time)

        # Extract categories
        rec_categories = []
        for rec in recs:
            isbn = rec.page_content.strip('"').split()[0]
            book_match = books[books["isbn13"] == isbn]
            if not book_match.empty:
                cat = book_match.iloc[0]["simple_category"]
                rec_categories.append(cat)

        if not rec_categories:
            continue

        # Prediction = Majority Vote
        from collections import Counter
        most_common = Counter(rec_categories).most_common(1)
        predicted_category = most_common[0][0]

        y_true.append(ground_truth_category)
        y_pred.append(predicted_category)

        # Precision@K
        matches = [1 if c == ground_truth_category else 0 for c in rec_categories]
        match_count = sum(matches)
        correct_retrievals += match_count
        total_retrievals += len(rec_categories)

        # MRR
        try:
            first_match_idx = matches.index(1)
            mrr_sum += 1.0 / (first_match_idx + 1)
        except ValueError:
            pass

        # Hit Rate
        if match_count > 0:
            hit_rate_sum += 1

        # NDCG
        dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(matches)])
        ideal_matches = sorted(matches, reverse=True)
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_matches)])

        if idcg > 0:
            ndcg_sum += dcg / idcg

    # Calculate metrics
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    avg_precision_at_k = correct_retrievals / total_retrievals if total_retrievals > 0 else 0
    mrr = mrr_sum / len(sample)
    hit_rate = hit_rate_sum / len(sample)
    ndcg = ndcg_sum / len(sample)
    avg_inference_time = np.mean(inference_times)

    metrics = {
        'model': model_name,
        'load_time': load_time,
        'avg_inference_time': avg_inference_time,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'precision_at_k': avg_precision_at_k,
        'mrr': mrr,
        'hit_rate': hit_rate,
        'ndcg': ndcg,
        'sample_size': len(sample)
    }

    print(f"Load Time: {load_time:.2f}s")
    print(f"Avg Inference Time: {avg_inference_time:.4f}s")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Precision@{top_k}: {avg_precision_at_k:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Hit Rate@{top_k}: {hit_rate:.4f}")
    print(f"NDCG@{top_k}: {ndcg:.4f}")

    return metrics

def main():
    # Models to compare
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",  # Current model
        "sentence-transformers/all-MiniLM-L12-v2", # Larger MiniLM
        "sentence-transformers/all-mpnet-base-v2", # Better performance
        "sentence-transformers/all-distilroberta-v1", # Efficient RoBERTa
        "sentence-transformers/paraphrase-MiniLM-L6-v2" # Paraphrase variant
    ]

    results = []

    for model in models:
        try:
            metrics = benchmark_model(model, sample_size=50, top_k=5)
            results.append(metrics)
        except Exception as e:
            print(f"Error benchmarking {model}: {e}")
            results.append({'model': model, 'error': str(e)})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("model_comparison_results.csv", index=False)
    print("\nResults saved to model_comparison_results.csv")

    # Print summary
    print("\n--- Model Comparison Summary ---")
    summary_cols = ['model', 'load_time', 'avg_inference_time', 'weighted_f1', 'precision_at_k', 'mrr', 'hit_rate', 'ndcg']
    print(df[summary_cols].to_string(index=False))

if __name__ == "__main__":
    main()
