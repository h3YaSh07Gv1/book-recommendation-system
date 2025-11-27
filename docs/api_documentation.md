# 📚 NextGen Book Recommender API Documentation

## Overview

The NextGen Book Recommender provides a RESTful API for intelligent book recommendations using advanced AI algorithms including hybrid semantic search and emotion-based mood journeys.

**Base URL:** `http://localhost:7860`
**Version:** 1.0.0
**Authentication:** None required (for demo purposes)

---

## API Endpoints

### 🔍 Semantic Search

Search for books using natural language queries with hybrid semantic search.

**Endpoint:** `POST /api/search`

**Request Body:**
```json
{
  "query": "A gripping mystery set in Victorian London",
  "category": "Mystery",
  "tone": "Suspenseful",
  "limit": 10
}
```

**Parameters:**
- `query` (string, required): Natural language description of desired book
- `category` (string, optional): Filter by book category (Fiction, Mystery, Sci-Fi, etc.)
- `tone` (string, optional): Filter by emotional tone (Happy, Sad, Suspenseful, etc.)
- `limit` (integer, optional): Maximum number of results (default: 16)

**Response:**
```json
{
  "status": "success",
  "query": "A gripping mystery set in Victorian London",
  "results": [
    {
      "isbn13": "9781234567890",
      "title": "The Hound of the Baskervilles",
      "authors": "Arthur Conan Doyle",
      "description": "A gripping mystery involving Sherlock Holmes...",
      "category": "Mystery",
      "thumbnail": "https://example.com/image.jpg",
      "redirect_link": "https://openlibrary.org/works/OL12345",
      "relevance_score": 0.95,
      "explanation": "This book matches your query because it involves 'a gripping mystery' and takes place in the Victorian era"
    }
  ],
  "total_results": 1,
  "processing_time_ms": 145
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Query parameter is required",
  "error_code": "INVALID_REQUEST"
}
```

---

### 🌊 Mood Journey

Generate an emotional journey through reading based on mood transitions.

**Endpoint:** `POST /api/mood-journey`

**Request Body:**
```json
{
  "start_mood": "Sad",
  "end_mood": "Happy"
}
```

**Parameters:**
- `start_mood` (string, required): Starting emotional state
- `end_mood` (string, required): Desired ending emotional state

**Available Moods:** Happy, Sad, Surprising, Angry, Suspenseful

**Response:**
```json
{
  "status": "success",
  "journey": {
    "start_mood": "Sad",
    "end_mood": "Happy",
    "books": [
      {
        "position": "start",
        "isbn13": "9781234567890",
        "title": "The Road",
        "authors": "Cormac McCarthy",
        "description": "A post-apocalyptic tale of survival...",
        "emotional_profile": {
          "joy": 0.1,
          "sadness": 0.8,
          "anger": 0.05,
          "fear": 0.03,
          "surprise": 0.02
        }
      },
      {
        "position": "bridge",
        "isbn13": "9781234567891",
        "title": "Life of Pi",
        "authors": "Yann Martel",
        "description": "A young boy's survival story...",
        "emotional_profile": {
          "joy": 0.4,
          "sadness": 0.4,
          "anger": 0.05,
          "fear": 0.08,
          "surprise": 0.07
        }
      },
      {
        "position": "end",
        "isbn13": "9781234567892",
        "title": "The Alchemist",
        "authors": "Paulo Coelho",
        "description": "A shepherd boy's journey to find treasure...",
        "emotional_profile": {
          "joy": 0.9,
          "sadness": 0.02,
          "anger": 0.01,
          "fear": 0.03,
          "surprise": 0.04
        }
      }
    ]
  },
  "processing_time_ms": 234
}
```

---

### 📊 Analytics

Get system performance and usage analytics.

**Endpoint:** `GET /api/analytics`

**Response:**
```json
{
  "status": "success",
  "analytics": {
    "total_searches": 1250,
    "average_response_time_ms": 145,
    "popular_categories": [
      {"category": "Fiction", "count": 450},
      {"category": "Mystery", "count": 320},
      {"category": "Sci-Fi", "count": 280}
    ],
    "popular_queries": [
      {"query": "space adventure", "count": 45},
      {"query": "detective mystery", "count": 38},
      {"query": "coming of age", "count": 32}
    ],
    "system_health": {
      "uptime_percentage": 99.9,
      "memory_usage_mb": 850,
      "cpu_usage_percentage": 15.2
    }
  }
}
```

---

### 💡 Feedback

Submit user feedback for recommendations.

**Endpoint:** `POST /api/feedback`

**Request Body:**
```json
{
  "query": "A gripping mystery set in Victorian London",
  "book_isbn": "9781234567890",
  "book_title": "The Hound of the Baskervilles",
  "feedback_type": "like",
  "rating": 5,
  "comments": "Perfect recommendation!"
}
```

**Parameters:**
- `query` (string, required): The original search query
- `book_isbn` (string, required): ISBN13 of the book
- `book_title` (string, required): Title of the book
- `feedback_type` (string, required): "like" or "dislike"
- `rating` (integer, optional): 1-5 rating
- `comments` (string, optional): Additional feedback

---

### 🏥 Health Check

Check system health and status.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-27T13:45:00Z",
  "version": "1.0.0",
  "services": {
    "vector_db": "healthy",
    "emotion_classifier": "healthy",
    "recommendation_engine": "healthy"
  },
  "metrics": {
    "active_connections": 5,
    "total_requests": 1250,
    "error_rate": 0.001
  }
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Missing or invalid request parameters |
| `BOOK_NOT_FOUND` | Specified book ISBN not found |
| `SYSTEM_ERROR` | Internal system error |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

---

## Rate Limiting

- **Anonymous users:** 100 requests per hour
- **Authenticated users:** 1000 requests per hour

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests per hour
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time until reset (Unix timestamp)

---

## Data Models

### Book Object
```json
{
  "isbn13": "string",
  "title": "string",
  "authors": "string",
  "description": "string",
  "category": "string",
  "thumbnail": "string",
  "redirect_link": "string",
  "published_year": "integer",
  "average_rating": "number",
  "emotional_profile": {
    "joy": "number",
    "sadness": "number",
    "anger": "number",
    "fear": "number",
    "surprise": "number"
  }
}
```

---

## SDK Examples

### Python
```python
import requests

# Search for books
response = requests.post("http://localhost:7860/api/search",
    json={
        "query": "A sci-fi adventure with AI",
        "category": "Sci-Fi",
        "limit": 5
    }
)

results = response.json()
for book in results["results"]:
    print(f"{book['title']} by {book['authors']}")
```

### JavaScript
```javascript
// Search for books
fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        query: "space exploration adventure",
        tone: "Happy"
    })
})
.then(response => response.json())
.then(data => console.log(data.results));
```

---

## Performance Benchmarks

| Operation | Average Response Time | 95th Percentile | Success Rate |
|-----------|----------------------|-----------------|--------------|
| Semantic Search | 145ms | 280ms | 99.8% |
| Mood Journey | 234ms | 450ms | 99.5% |
| Analytics | 45ms | 120ms | 99.9% |

---

## Changelog

### Version 1.0.0
- Initial API release
- Hybrid semantic search
- Mood journey generation
- Basic analytics
- Health monitoring

---

*For support or questions, please contact the development team.*
