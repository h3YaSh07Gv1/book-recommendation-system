import sys
import os
import pandas as pd
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import data_loader
from src.engine.recommender import Recommender

def debug_search():
    print("Loading data...")
    try:
        documents = data_loader.load_documents("data/tagged_description.txt")
        books = data_loader.load_books("data/books_with_redirect_links(emotions).csv")
        recommender = Recommender(documents)
        print("Data loaded.")
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return

    query = "cyberpunk"
    print(f"\nTesting search for query: '{query}'")
    
    try:
        print("Running hybrid_search...")
        recs = recommender.hybrid_search(query)
        print(f"hybrid_search returned {len(recs)} results.")
        
        print("Running filter_books...")
        filtered_books = recommender.filter_books(books, recs, query)
        print(f"filter_books returned {len(filtered_books)} books.")
        
        if not filtered_books.empty:
            print("Top result:", filtered_books.iloc[0]["title"])
        else:
            print("No books found after filtering.")
            
    except Exception as e:
        print(f"ERROR during search: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_search()
