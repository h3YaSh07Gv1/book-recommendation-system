import pandas as pd
import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def ingest_data(csv_path, output_dir="data"):
    """
    Ingests a new CSV of books.
    Expected columns: title, description, authors, thumbnail, simple_category, etc.
    """
    print(f"Reading {csv_path}...")
    try:
        new_books = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Basic validation
    required_cols = ["title", "description", "authors"]
    for col in required_cols:
        if col not in new_books.columns:
            print(f"Missing required column: {col}")
            return

    # 1. Update Books CSV
    # Load existing
    existing_csv = os.path.join(output_dir, "books_with_redirect_links(emotions).csv")
    if os.path.exists(existing_csv):
        existing_books = pd.read_csv(existing_csv)
        # Append
        combined_books = pd.concat([existing_books, new_books], ignore_index=True)
        # Deduplicate by ISBN if exists, else title
        if "isbn13" in combined_books.columns:
            combined_books.drop_duplicates(subset="isbn13", inplace=True)
        else:
            combined_books.drop_duplicates(subset="title", inplace=True)
            
        combined_books.to_csv(existing_csv, index=False)
        print(f"Updated {existing_csv}. Total books: {len(combined_books)}")
    else:
        print("Existing books CSV not found.")

    # 2. Update Vector DB
    # We need to create a text file for the loader or just add texts directly to Chroma
    # The current system uses 'tagged_description.txt'. We should append to it.
    
    txt_path = os.path.join(output_dir, "tagged_description.txt")
    
    with open(txt_path, "a", encoding="utf-8") as f:
        for _, row in new_books.iterrows():
            # Format: ISBN title category description
            # Note: The original format seems to be just lines of text? 
            # Let's check the loader. It uses TextLoader.
            # We need to match the format exactly if we want the ID extraction to work.
            # In recommender.py: int(rec.page_content.strip('"').split()[0])
            # So the first token MUST be the ISBN (or ID).
            
            isbn = row.get("isbn13", row.get("isbn", "000000"))
            line = f"{isbn} {row['title']} {row.get('simple_category', 'Unknown')} {row['description']}\n"
            f.write(line)
            
    print(f"Appended descriptions to {txt_path}")
    
    # Re-embed?
    # Since we are using a local Chroma instance that might be transient or persistent.
    # The current code initializes Chroma from documents every time app starts.
    # So just updating the txt file is enough for the NEXT restart.
    print("Ingestion complete. Restart the app to see changes.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_data.py <path_to_new_books.csv>")
    else:
        ingest_data(sys.argv[1])
