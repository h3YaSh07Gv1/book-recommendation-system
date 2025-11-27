import requests
import pandas as pd
import argparse
import time
import os

def fetch_books_by_subject(subject, limit=100):
    """
    Fetches books from OpenLibrary by subject.
    """
    print(f"Fetching up to {limit} books for subject: {subject}...")
    url = f"https://openlibrary.org/search.json?subject={subject}&limit={limit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching search results: {e}")
        return None

    books = []
    docs = data.get("docs", [])
    
    print(f"Found {len(docs)} initial results. Fetching details...")
    
    for i, doc in enumerate(docs):
        # Basic info
        title = doc.get("title")
        key = doc.get("key") # /works/OL123W
        authors = doc.get("author_name", ["Unknown"])
        authors_str = ";".join(authors)
        cover_id = doc.get("cover_i")
        
        if not key:
            continue
            
        # We need description, which is often not in search results.
        # We must fetch the work details.
        # Rate limiting: OpenLibrary asks for 1 req/sec roughly if not authenticated? 
        # Let's be polite.
        
        work_url = f"https://openlibrary.org{key}.json"
        try:
            time.sleep(0.5) # Be polite
            work_resp = requests.get(work_url)
            if work_resp.status_code != 200:
                continue
            work_data = work_resp.json()
            
            description = work_data.get("description")
            if isinstance(description, dict):
                description = description.get("value")
            
            if not description:
                # Skip books without description as they are useless for semantic search
                print(f"Skipping '{title}' (no description)")
                continue
                
            # Construct row
            # Columns expected: isbn13,isbn10,title,subtitle,authors,categories,thumbnail,description,published_year,average_rating,num_pages,ratings_count,redirect_link,simple_category
            
            # Thumbnail
            thumbnail = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else ""
            
            book = {
                "isbn13": f"OL_{key.split('/')[-1]}", # Fake ISBN using OL Key
                "title": title,
                "authors": authors_str,
                "description": description,
                "simple_category": subject.capitalize(), # Use subject as category
                "thumbnail": thumbnail,
                "redirect_link": f"https://openlibrary.org{key}",
                "average_rating": 0, # Placeholder
                "published_year": doc.get("first_publish_year", 0)
            }
            
            books.append(book)
            print(f"Fetched: {title}")
            
        except Exception as e:
            print(f"Error fetching work details for {key}: {e}")
            continue
            
    return pd.DataFrame(books)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch books from OpenLibrary")
    parser.add_argument("--subject", type=str, required=True, help="Subject to search for (e.g., 'cyberpunk')")
    parser.add_argument("--limit", type=int, default=10, help="Number of books to check (default 10)")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    df = fetch_books_by_subject(args.subject, args.limit)
    
    if df is not None and not df.empty:
        output_file = os.path.join(args.output, f"new_books_{args.subject}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} books to {output_file}")
    else:
        print("No books found or saved.")
