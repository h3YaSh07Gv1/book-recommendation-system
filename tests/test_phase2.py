import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import data_loader
from src.engine.recommender import Recommender
from src.data.feedback_store import feedback_store

def test_phase2():
    print("Loading data...")
    documents = data_loader.load_documents("data/tagged_description.txt")
    books = data_loader.load_books("data/books_with_redirect_links(emotions).csv")
    recommender = Recommender(documents)
    print("Data loaded.")

    print("\n--- Test 1: Feedback Loop ---")
    query = "love story"
    
    # 1. Get initial results
    print("Initial Search:")
    recs = recommender.semantic_search(query, k=10)
    initial_results = recommender.filter_books(books, recs, query, top_k=5)
    top_book = initial_results.iloc[0]["title"]
    second_book = initial_results.iloc[1]["title"]
    print(f"Top book: {top_book}")
    print(f"Second book: {second_book}")
    
    # 2. Give feedback: Like the second book
    print(f"Liking '{second_book}'...")
    feedback_store.log_feedback(query, second_book, "like")
    
    # 3. Search again
    print("Search after feedback:")
    recs_new = recommender.semantic_search(query, k=10)
    new_results = recommender.filter_books(books, recs_new, query, top_k=5)
    new_top_book = new_results.iloc[0]["title"]
    print(f"New Top book: {new_top_book}")
    
    if new_top_book == second_book:
        print("SUCCESS: Liked book moved to top!")
    else:
        print(f"Note: Liked book rank: {new_results[new_results['title'] == second_book].index[0] if second_book in new_results['title'].values else 'Not in top 5'}")

    print("\n--- Test 2: Improved Mood Journey ---")
    start = "Sad"
    end = "Happy"
    journey = recommender.get_mood_journey(books, start, end)
    if len(journey) == 3:
        print(f"Start ({start}): {journey[0]['title']}")
        print(f"Bridge (Interpolated): {journey[1]['title']}")
        print(f"End ({end}): {journey[2]['title']}")
        print("SUCCESS: Journey generated.")
    else:
        print("FAILED: Journey generation.")

if __name__ == "__main__":
    test_phase2()
