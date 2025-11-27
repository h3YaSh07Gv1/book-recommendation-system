import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from src.data.feedback_store import feedback_store
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class Recommender:
    def __init__(self, documents, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        # Create unique collection name based on model to avoid dimension conflicts
        import hashlib
        model_hash = hashlib.md5(embedding_model_name.encode()).hexdigest()[:8]
        collection_name = f"books_{model_hash}"
        self.db_books = Chroma.from_documents(documents, embedding=self.embedding_model, collection_name=collection_name)
        self.documents = documents

        # Initialize BM25
        print("Initializing BM25...")
        self.corpus_tokens = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        # Initialize Cross-Encoder (lazy load if possible, but we'll load now)
        print("Initializing Cross-Encoder...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Pre-compute emotion vectors for faster mood journey
        self.emotion_cols = ["joy", "surprise", "anger", "fear", "sadness"]

    def semantic_search(self, query: str, k: int = 50) -> List[Document]:
        """Performs semantic search using ChromaDB."""
        recs = self.db_books.similarity_search(query, k=k)
        return recs

    def hybrid_search(self, query: str, k: int = 50) -> List[Document]:
        """
        Combines BM25 and Vector Search using Reciprocal Rank Fusion (RRF).
        Then re-ranks top results using Cross-Encoder.
        """
        # 1. Vector Search
        vector_recs = self.db_books.similarity_search(query, k=k)
        vector_isbns = [rec.page_content.strip('"').split()[0] for rec in vector_recs]
        
        # 2. BM25 Search
        tokenized_query = query.lower().split()
        bm25_docs = self.bm25.get_top_n(tokenized_query, self.documents, n=k)
        bm25_isbns = [doc.page_content.strip('"').split()[0] for doc in bm25_docs]
        
        # 3. RRF Fusion
        # Score = 1 / (k + rank)
        rrf_score = {}
        for rank, isbn in enumerate(vector_isbns):
            rrf_score[isbn] = rrf_score.get(isbn, 0) + (1 / (60 + rank))
            
        for rank, isbn in enumerate(bm25_isbns):
            rrf_score[isbn] = rrf_score.get(isbn, 0) + (1 / (60 + rank))
            
        # Sort by RRF score
        sorted_isbns = sorted(rrf_score.keys(), key=lambda x: rrf_score[x], reverse=True)
        top_candidates_isbns = sorted_isbns[:20] # Take top 20 for re-ranking
        
        # 4. Cross-Encoder Re-ranking
        # We need the text content for these ISBNs
        # Map ISBN to Document
        isbn_to_doc = {}
        # This is inefficient, iterating all docs. Better to have a map.
        # But for 6k docs it's okay-ish. Optimization: Create map in __init__
        # Let's do a quick lookup map creation on init or just iterate now.
        # Actually, we have the docs from the search results!
        
        candidate_docs = []
        # Collect docs from vector and bm25 results that match the top candidates
        # This might miss some if they weren't in the top K of both? No, they are from the union.
        
        # Let's just build a quick map from the search results we already have
        found_docs = {}
        for doc in vector_recs:
            isbn = doc.page_content.strip('"').split()[0]
            found_docs[isbn] = doc.page_content
        for doc in bm25_docs:
            isbn = doc.page_content.strip('"').split()[0]
            found_docs[isbn] = doc.page_content
            
        pairs = []
        valid_isbns = []
        for isbn in top_candidates_isbns:
            if isbn in found_docs:
                # Content format: "ISBN Title Category Description"
                # We want to re-rank based on the description part mainly?
                # The CrossEncoder expects (query, document_text)
                pairs.append([query, found_docs[isbn]])
                valid_isbns.append(isbn)
                
        if not pairs:
            return vector_recs[:k] # Fallback
            
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by Cross-Encoder score
        scored_candidates = sorted(zip(valid_isbns, scores), key=lambda x: x[1], reverse=True)
        
        # Return as list of "Document" objects (mocked or retrieved)
        # We need to return objects that look like Chroma results for compatibility
        final_recs = []
        for isbn, score in scored_candidates:
            # We need to reconstruct the Document object or just pass the content
            # The downstream filter_books expects objects with page_content
            from langchain_core.documents import Document
            final_recs.append(Document(page_content=found_docs[isbn]))
            
        return final_recs

    def filter_books(self, books_df: pd.DataFrame, recs: List[Document], query: str, category: Optional[str] = None, tone: Optional[str] = None, top_k: int = 16) -> pd.DataFrame:
        """Filters recommendations based on category and tone, and re-ranks based on feedback."""
        if not recs:
            return pd.DataFrame()
            
        required_cols = ["isbn13", "simple_category", "title", "large_thumbnail"]
        missing_cols = [col for col in required_cols if col not in books_df.columns]
        if missing_cols:
            print(f"Warning: books_df missing columns: {missing_cols}")
            # We can still try to proceed if critical cols like isbn13 are there
            if "isbn13" not in books_df.columns:
                return pd.DataFrame()
            
        books_list = [rec.page_content.strip('"').split()[0] for rec in recs]
        
        # Create a DataFrame from the recommendations to preserve order/score if possible
        # Chroma returns list, so order is by similarity.
        # We need to map this order to the dataframe.
        
        # Ensure isbn13 in books_df is string for comparison
        books_df["isbn13"] = books_df["isbn13"].astype(str)
        books_recs = books_df[books_df["isbn13"].isin(books_list)].copy()
        
        # Re-rank based on Feedback
        # We'll add a 'score' column. Initial score is based on reverse index in the search results (higher is better)
        # But since we lost the exact similarity score from Chroma (unless we use similarity_search_with_score),
        # we'll just assume the order in 'books_list' is the relevance.
        
        # Map ISBN to rank
        isbn_rank = {isbn: i for i, isbn in enumerate(books_list)}
        books_recs["base_rank"] = books_recs["isbn13"].map(isbn_rank)
        
        # Calculate feedback score for each book
        books_recs["feedback_score"] = books_recs["title"].apply(lambda t: feedback_store.get_feedback_score(query, t))
        
        # Final Score = (Max Rank - Base Rank) + (Feedback Score * Weight)
        # Lower base_rank is better.
        max_rank = len(books_list)
        books_recs["final_score"] = (max_rank - books_recs["base_rank"]) + (books_recs["feedback_score"] * 5)
        
        # Sort by final score
        books_recs.sort_values(by="final_score", ascending=False, inplace=True)

        if category and category != "All":
            books_recs = books_recs[books_recs["simple_category"] == category]
        
        # Tone sorting (secondary sort)
        if tone and tone != "All":
             tone_map = {
                 "Happy": "joy",
                 "Surprising": "surprise",
                 "Angry": "anger",
                 "Suspenseful": "fear",
                 "Sad": "sadness"
             }
             if tone in tone_map:
                 # We want to keep the semantic relevance but also respect the tone.
                 # Let's add tone value to the score?
                 # Or just sort by tone within the top results?
                 # Let's just re-sort the top results by tone for now as requested originally.
                 books_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)
        
        return books_recs.head(top_k)

    def get_mood_journey(self, books_df: pd.DataFrame, start_mood: str, end_mood: str) -> List[Any]:
        """
        Returns a sequence of 3 books transitioning from start_mood to end_mood.
        Uses vector interpolation for the bridge book.
        """
        tone_map = {
             "Happy": "joy",
             "Surprising": "surprise",
             "Angry": "anger",
             "Suspenseful": "fear",
             "Sad": "sadness"
        }
        
        if start_mood not in tone_map or end_mood not in tone_map:
            return []

        # 1. Start Book: Randomly pick from top 20 high in start_mood
        start_candidates = books_df.sort_values(by=tone_map[start_mood], ascending=False).head(20)
        start_book = start_candidates.sample(n=1).iloc[0]
        
        # 2. End Book: Randomly pick from top 20 high in end_mood
        end_candidates = books_df.sort_values(by=tone_map[end_mood], ascending=False).head(20)
        end_book = end_candidates.sample(n=1).iloc[0]
        
        # 3. Bridge Book: Vector Interpolation
        # We want a book that represents the "midpoint" between the two emotions.
        # We will look at the 5 emotion columns as a vector.
        
        start_vec = start_book[self.emotion_cols].values.astype(float)
        end_vec = end_book[self.emotion_cols].values.astype(float)
        
        target_vec = (start_vec + end_vec) / 2
        
        # Calculate distance of all books to this target vector
        # This is expensive to do on the fly for all books (6k).
        # But 6k is small enough for numpy.
        
        all_vecs = books_df[self.emotion_cols].values.astype(float)
        
        # Euclidean distance
        dists = np.linalg.norm(all_vecs - target_vec, axis=1)
        
        # Find book with min distance that is NOT start or end book
        books_df["temp_dist"] = dists
        sorted_books = books_df.sort_values("temp_dist")
        
        # Filter out start/end books and take top 10 candidates for variety
        candidates = sorted_books[
            (sorted_books["isbn13"] != start_book["isbn13"]) & 
            (sorted_books["isbn13"] != end_book["isbn13"])
        ].head(10)
        
        if candidates.empty:
             bridge_book = sorted_books.iloc[2] # Fallback
        else:
             bridge_book = candidates.sample(n=1).iloc[0]

        return [start_book, bridge_book, end_book]

    def explain_match(self, query: str, book_description: str) -> str:
        """
        Identifies the sentence in the book description that best matches the query.
        """
        sentences = book_description.split('.')
        sentences = [s.strip() for s in sentences if len(s) > 10] # Filter short junk
        
        if not sentences:
            return "No description available to analyze."

        # Embed query and sentences
        # Note: This might be slow if we do it for every result on the fly. 
        # Optimization: Only do it for the top 1-3 results or on demand.
        # For now, let's just do it for the top result or just return the first sentence if too slow.
        
        # To keep it fast/resource-light, we can use simple Jaccard similarity or word overlap 
        # if we don't want to run the embedding model 20 times per request.
        # But we have the embedding model loaded, so let's use it but sparingly.
        
        # Let's try a simple keyword overlap first for speed.
        query_words = set(query.lower().split())
        best_score = 0
        best_sentence = sentences[0]
        
        for sent in sentences:
            sent_words = set(sent.lower().split())
            if not sent_words: continue
            overlap = len(query_words.intersection(sent_words))
            if overlap > best_score:
                best_score = overlap
                best_sentence = sent
        
        return best_sentence
