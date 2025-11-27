import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.engine.recommender import Recommender
from src.data.loader import DataLoader


class TestRecommender:
    """Comprehensive test suite for the Recommender class."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Mock(page_content="9781234567890 Test Book 1 Fiction A great adventure story about heroes and villains."),
            Mock(page_content="9781234567891 Test Book 2 Mystery A thrilling mystery with unexpected twists."),
            Mock(page_content="9781234567892 Test Book 3 Sci-Fi A futuristic tale of technology and humanity.")
        ]

    @pytest.fixture
    def sample_books_df(self):
        """Create sample books dataframe for testing."""
        return pd.DataFrame({
            'isbn13': ['9781234567890', '9781234567891', '9781234567892'],
            'title': ['Test Book 1', 'Test Book 2', 'Test Book 3'],
            'authors': ['Author One', 'Author Two', 'Author Three'],
            'description': [
                'A great adventure story about heroes and villains.',
                'A thrilling mystery with unexpected twists.',
                'A futuristic tale of technology and humanity.'
            ],
            'large_thumbnail': ['image1.jpg', 'image2.jpg', 'image3.jpg'],
            'simple_category': ['Fiction', 'Mystery', 'Sci-Fi'],
            'joy': [0.8, 0.3, 0.6],
            'sadness': [0.1, 0.2, 0.1],
            'anger': [0.05, 0.1, 0.1],
            'fear': [0.03, 0.3, 0.1],
            'surprise': [0.02, 0.1, 0.1]
        })

    @patch('src.engine.recommender.HuggingFaceEmbeddings')
    @patch('src.engine.recommender.Chroma')
    @patch('src.engine.recommender.BM25Okapi')
    @patch('src.engine.recommender.CrossEncoder')
    def test_recommender_initialization(self, mock_cross_encoder, mock_bm25, mock_chroma, mock_embeddings, sample_documents):
        """Test recommender initialization."""
        # Setup mocks
        mock_chroma.from_documents.return_value = Mock()
        mock_bm25.return_value = Mock()
        mock_cross_encoder.return_value = Mock()

        # Initialize recommender
        recommender = Recommender(sample_documents)

        # Assertions
        assert recommender.documents == sample_documents
        assert recommender.emotion_cols == ["joy", "surprise", "anger", "fear", "sadness"]
        mock_embeddings.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")
        mock_chroma.from_documents.assert_called_once()
        mock_bm25.assert_called_once()
        mock_cross_encoder.assert_called_once_with('cross-encoder/ms-marco-MiniLM-L-6-v2')

    @patch('src.engine.recommender.Chroma')
    def test_semantic_search(self, mock_chroma_class, sample_documents):
        """Test semantic search functionality."""
        # Setup
        mock_db = Mock()
        mock_chroma_class.from_documents.return_value = mock_db
        mock_db.similarity_search.return_value = [sample_documents[0], sample_documents[1]]

        with patch('src.engine.recommender.HuggingFaceEmbeddings'), \
             patch('src.engine.recommender.BM25Okapi'), \
             patch('src.engine.recommender.CrossEncoder'):
            recommender = Recommender(sample_documents)

        # Execute
        results = recommender.semantic_search("adventure story", k=2)

        # Assertions
        mock_db.similarity_search.assert_called_once_with("adventure story", k=2)
        assert len(results) == 2
        assert results == [sample_documents[0], sample_documents[1]]

    def test_hybrid_search_basic(self, sample_documents, sample_books_df):
        """Test basic hybrid search functionality."""
        with patch('src.engine.recommender.HuggingFaceEmbeddings'), \
             patch('src.engine.recommender.Chroma') as mock_chroma_class, \
             patch('src.engine.recommender.BM25Okapi'), \
             patch('src.engine.recommender.CrossEncoder') as mock_cross_encoder_class:

            # Setup mocks
            mock_db = Mock()
            mock_chroma_class.from_documents.return_value = mock_db
            mock_db.similarity_search.return_value = sample_documents[:2]

            mock_bm25 = Mock()
            mock_bm25.get_top_n.return_value = sample_documents[1:]

            mock_cross_encoder = Mock()
            mock_cross_encoder.predict.return_value = [0.9, 0.8]

            mock_chroma_class.from_documents.return_value = mock_db
            mock_bm25_instance = Mock()
            mock_bm25.return_value = mock_bm25_instance
            mock_bm25_instance.get_top_n.return_value = sample_documents[1:]
            mock_cross_encoder_class.return_value = mock_cross_encoder

            recommender = Recommender(sample_documents)

            # Execute
            results = recommender.hybrid_search("test query")

            # Assertions
            assert len(results) == 2
            mock_db.similarity_search.assert_called()
            mock_bm25_instance.get_top_n.assert_called()
            mock_cross_encoder.predict.assert_called()

    def test_filter_books_with_category(self, sample_documents, sample_books_df):
        """Test book filtering with category."""
        with patch('src.engine.recommender.HuggingFaceEmbeddings'), \
             patch('src.engine.recommender.Chroma'), \
             patch('src.engine.recommender.BM25Okapi'), \
             patch('src.engine.recommender.CrossEncoder'):

            recommender = Recommender(sample_documents)

            # Mock recs
            mock_recs = [
                Mock(page_content="9781234567890 Test Book 1 Fiction Description"),
                Mock(page_content="9781234567891 Test Book 2 Mystery Description")
            ]

            # Execute
            filtered = recommender.filter_books(sample_books_df, mock_recs, "test query", category="Fiction")

            # Assertions
            assert len(filtered) == 1
            assert filtered.iloc[0]['simple_category'] == 'Fiction'

    def test_filter_books_with_tone(self, sample_documents, sample_books_df):
        """Test book filtering with emotional tone."""
        with patch('src.engine.recommender.HuggingFaceEmbeddings'), \
             patch('src.engine.recommender.Chroma'), \
             patch('src.engine.recommender.BM25Okapi'), \
             patch('src.engine.recommender.CrossEncoder'):

            recommender = Recommender(sample_documents)

            mock_recs = [
                Mock(page_content="9781234567890 Test Book 1 Fiction Description"),
                Mock(page_content="9781234567891 Test Book 2 Mystery Description"),
                Mock(page_content="9781234567892 Test Book 3 Sci-Fi Description")
            ]

            # Execute with Happy tone (joy)
            filtered = recommender.filter_books(sample_books_df, mock_recs, "test query", tone="Happy")

            # Should be sorted by joy score descending
            assert filtered.iloc[0]['joy'] >= filtered.iloc[1]['joy']

    def test_mood_journey_generation(self, sample_documents, sample_books_df):
        """Test mood journey generation."""
        with patch('src.engine.recommender.HuggingFaceEmbeddings'), \
             patch('src.engine.recommender.Chroma'), \
             patch('src.engine.recommender.BM25Okapi'), \
             patch('src.engine.recommender.CrossEncoder'):

            recommender = Recommender(sample_documents)

            # Execute
            journey = recommender.get_mood_journey(sample_books_df, "Sad", "Happy")

            # Assertions
            assert len(journey) == 3  # start, bridge, end
            assert journey[0] is not None
            assert journey[1] is not None
            assert journey[2] is not None

    def test_explain_match_functionality(self, sample_documents):
        """Test match explanation functionality."""
        with patch('src.engine.recommender.HuggingFaceEmbeddings'), \
             patch('src.engine.recommender.Chroma'), \
             patch('src.engine.recommender.BM25Okapi'), \
             patch('src.engine.recommender.CrossEncoder'):

            recommender = Recommender(sample_documents)

            description = "A great adventure story about heroes and villains in a magical world."
            query = "adventure heroes"

            # Execute
            explanation = recommender.explain_match(query, description)

            # Assertions
            assert isinstance(explanation, str)
            assert len(explanation) > 0

    @pytest.mark.parametrize("start_mood,end_mood", [
        ("Happy", "Sad"),
        ("Sad", "Happy"),
        ("Angry", "Surprising"),
        ("Suspenseful", "Happy")
    ])
    def test_mood_journey_different_combinations(self, sample_documents, sample_books_df, start_mood, end_mood):
        """Test mood journey with different emotional combinations."""
        with patch('src.engine.recommender.HuggingFaceEmbeddings'), \
             patch('src.engine.recommender.Chroma'), \
             patch('src.engine.recommender.BM25Okapi'), \
             patch('src.engine.recommender.CrossEncoder'):

            recommender = Recommender(sample_documents)

            journey = recommender.get_mood_journey(sample_books_df, start_mood, end_mood)

            assert len(journey) == 3
            assert all(book is not None for book in journey)

    def test_error_handling_invalid_mood(self, sample_documents, sample_books_df):
        """Test error handling for invalid mood inputs."""
        with patch('src.engine.recommender.HuggingFaceEmbeddings'), \
             patch('src.engine.recommender.Chroma'), \
             patch('src.engine.recommender.BM25Okapi'), \
             patch('src.engine.recommender.CrossEncoder'):

            recommender = Recommender(sample_documents)

            # Test with invalid mood
            journey = recommender.get_mood_journey(sample_books_df, "InvalidMood", "Happy")

            assert journey == []

    def test_empty_search_results(self, sample_documents):
        """Test handling of empty search results."""
        with patch('src.engine.recommender.HuggingFaceEmbeddings'), \
             patch('src.engine.recommender.Chroma') as mock_chroma_class, \
             patch('src.engine.recommender.BM25Okapi'), \
             patch('src.engine.recommender.CrossEncoder'):

            mock_db = Mock()
            mock_chroma_class.from_documents.return_value = mock_db
            mock_db.similarity_search.return_value = []

            recommender = Recommender(sample_documents)

            filtered = recommender.filter_books(pd.DataFrame(), [], "test query")

            assert filtered.empty

    @pytest.mark.performance
    def test_performance_semantic_search(self, sample_documents, benchmark):
        """Performance test for semantic search."""
        with patch('src.engine.recommender.HuggingFaceEmbeddings'), \
             patch('src.engine.recommender.Chroma') as mock_chroma_class, \
             patch('src.engine.recommender.BM25Okapi'), \
             patch('src.engine.recommender.CrossEncoder'):

            mock_db = Mock()
            mock_chroma_class.from_documents.return_value = mock_db
            mock_db.similarity_search.return_value = sample_documents

            recommender = Recommender(sample_documents)

            # Benchmark the search
            result = benchmark(recommender.semantic_search, "test query", 10)

            assert len(result) > 0
            assert result[0] is not None
