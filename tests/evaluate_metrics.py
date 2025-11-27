import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from collections import Counter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import data_loader
from src.engine.recommender import Recommender

def evaluate_recommender(sample_size=100, top_k=5):
    print("Loading data...")
    documents = data_loader.load_documents("data/tagged_description.txt")
    books = data_loader.load_books("data/books_with_redirect_links(emotions).csv")
    books["isbn13"] = books["isbn13"].astype(str)
    recommender = Recommender(documents)
    print("Data loaded.")

    # Filter books with valid categories
    valid_books = books[books["simple_category"].notna() & (books["simple_category"] != "Unknown")]
    
    # Stratified sampling if possible, else random
    try:
        sample = valid_books.groupby('simple_category', group_keys=False).apply(lambda x: x.sample(min(len(x), int(sample_size/len(valid_books['simple_category'].unique())) + 1)))
        # Clip to sample_size
        if len(sample) > sample_size:
            sample = sample.sample(sample_size)
    except:
        sample = valid_books.sample(min(sample_size, len(valid_books)))

    print(f"Evaluating on {len(sample)} books...")

    y_true = []
    y_pred = []
    
    correct_retrievals = 0
    total_retrievals = 0
    
    mrr_sum = 0
    hit_rate_sum = 0
    ndcg_sum = 0

    for _, row in sample.iterrows():
        # Construct query
        query = f"{row['title']}. {row['description'][:100]}"
        ground_truth_category = row["simple_category"]
        
        # Get recommendations
        recs = recommender.hybrid_search(query, k=top_k)
        
        # Extract categories of recommended books
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
        most_common = Counter(rec_categories).most_common(1)
        predicted_category = most_common[0][0]
        
        y_true.append(ground_truth_category)
        y_pred.append(predicted_category)
        
        # Precision@K for this query (how many of top_k are same category)
        matches = [1 if c == ground_truth_category else 0 for c in rec_categories]
        match_count = sum(matches)
        correct_retrievals += match_count
        total_retrievals += len(rec_categories)

        # MRR (Mean Reciprocal Rank)
        # Find first index where match is 1
        try:
            first_match_idx = matches.index(1)
            mrr_sum += 1.0 / (first_match_idx + 1)
        except ValueError:
            pass # No match in top K
            
        # Hit Rate @ K
        if match_count > 0:
            hit_rate_sum += 1
            
        # NDCG (Normalized Discounted Cumulative Gain)
        # IDCG (Ideal DCG) - assuming all top K should be relevant (or as many as possible)
        # Since we only have 1 ground truth item (the source book), but we are checking category match.
        # Let's assume "Relevant" means same category.
        # DCG = sum(rel_i / log2(i+2))
        dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(matches)])
        
        # Ideal DCG: If we had 'match_count' relevant items, they would be at the top.
        # But wait, we don't know how many relevant items exist in total in the DB for this category?
        # Actually we do, but for @K metric, IDCG is calculated based on the best possible ordering of the RETRIEVED items 
        # OR the best possible ordering of K relevant items if they exist.
        # Standard definition: IDCG is the DCG of the ideal ordering of the top-k results.
        # If we have 'match_count' relevant items in the top k, the ideal ordering puts them all at the top.
        ideal_matches = sorted(matches, reverse=True)
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_matches)])
        
        if idcg > 0:
            ndcg_sum += dcg / idcg
        else:
            # If no relevant items were retrieved, NDCG is 0. 
            # But if no relevant items EXIST (impossible here since source is relevant), it would be 1?
            # Here, if idcg is 0, it means no matches found. So NDCG is 0.
            pass

    # Metrics
    print("\n--- Performance Metrics ---")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    
    avg_precision_at_k = correct_retrievals / total_retrievals if total_retrievals > 0 else 0
    print(f"Average Precision@{top_k} (Category Match): {avg_precision_at_k:.4f}")
    
    # Advanced Metrics
    mrr = mrr_sum / len(sample)
    hit_rate = hit_rate_sum / len(sample)
    ndcg = ndcg_sum / len(sample)
    
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print(f"Hit Rate @ {top_k}: {hit_rate:.4f}")
    print(f"NDCG @ {top_k}: {ndcg:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true) | set(y_pred))))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=sorted(list(set(y_true) | set(y_pred))), 
                yticklabels=sorted(list(set(y_true) | set(y_pred))))
    plt.title("Confusion Matrix: Category Prediction")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nConfusion matrix saved to 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_recommender()
