# CinemAI - Advanced Movie Recommendation System

## Proof of Concept Overview

CinemAI is a sophisticated movie and TV show recommendation system that leverages cutting-edge AI technologies combined with traditional algorithms to provide personalized recommendations. This proof of concept demonstrates a FastAPI backend architecture using AI embeddings, statistical clustering, and GPT-4 for intelligent movie suggestions.

**Current Status**: Backend API implementation with comprehensive recommendation pipeline  
**Future Plans**: Frontend development and recommendation system enhancements

## Implementation Status

### **Currently Implemented**
- **FastAPI Backend**: Complete REST API with dependency injection
- **Recommendation Pipeline**: Full recommendation engine with AI embeddings, statistical clustering, and GPT-4
- **Database Layer**: PostgreSQL with pgvector for vector similarity search
- **Caching System**: Redis for performance optimization
- **External APIs**: TMDB integration for movie data

### **In Development**
- **Frontend Application**: Modern web interface (planned)
- **User Authentication**: User management system (planned)
- **Real-time Learning**: Dynamic user profile updates (planned)

### **Future Enhancements**
- **Advanced Recommendation Algorithms**: Collaborative filtering, content-based filtering
- **A/B Testing Framework**: Algorithm comparison and optimization
- **Analytics Dashboard**: User behavior insights and metrics
- **Mobile Application**: React Native frontend
- **Microservices Architecture**: Scalable service decomposition

## How It Works

### 1. User Profile Creation Process
When a user uploads their watch history:

- **Data Validation**: Validates and enriches movie data using field detection
- **High-Rated Filtering**: Focuses on movies with ratings > 4 for quality preferences
- **Text Embedding Generation**: Creates semantic embeddings from movie titles, descriptions, genres, and countries
- **Weighted Clustering**: Uses adaptive clustering based on dataset size with rating and recency weights:
  - **< 100 movies**: Single cluster (weighted average)
  - **100-500 movies**: K-means with silhouette score optimization (2-10 clusters)
  - **> 500 movies**: HDBSCAN with minimum cluster size of 10
- **Profile Storage**: Saves user profile with clusters to PostgreSQL for future use

### 2. User Profile Retrieval
The system retrieves an existing user profile from the database:

- **Profile Lookup**: Queries PostgreSQL with pgvector for existing user profile
- **Error Handling**: If no profile exists, raises an error requiring watch history upload
- **Profile Validation**: Ensures profile contains movies and clusters for recommendation generation

### 3. Movie Candidate Generation
Fetches potential movie recommendations from external sources:

- **TMDB API Integration**: Retrieves movies based on user filters (genres, countries, years)
- **Filter Application**: Applies dynamic filters for genres, countries, year ranges
- **Batch Processing**: Efficiently handles large candidate sets

### 4. Candidate Filtering & Embedding
Prepares candidate movies for similarity analysis:

- **Watched Movie Exclusion**: Uses fuzzy matching with 85% threshold to remove movies the user has already seen
- **Fast Path Filtering**: Quick exact title matching for efficiency
- **Fuzzy Filtering**: Advanced string similarity with year tolerance (±1 year)
- **Additional Filters**: Applies genre, year, duration, and country filters
- **Deduplication**: Removes duplicate movies based on title and year
- **Text Representation**: Creates comprehensive text descriptions combining title, description, genres, and countries
- **Embedding Generation**: Uses OpenAI embeddings to convert text to high-dimensional vectors

### 5. Recommendation Engine
Core algorithm that matches user preferences with candidates:

- **Cluster-Based Matching**: Uses user's preference clusters to find similar movie groups
- **Cosine Similarity**: Computes similarity scores between user cluster centroids and candidate movies
- **Softmax Aggregation (New)**: Instead of using only the maximum similarity, the system now aggregates similarities across all clusters using a softmax-weighted sum. This provides a more nuanced score that considers all facets of user taste.
    - **Alpha (α) Parameter**: Controls the sharpness of the softmax weighting. Higher alpha makes the score focus more on the best-matching cluster (like max), while lower alpha spreads weight across all clusters (like mean). The default is α = 5.0, so it's something in between.
- **Top Candidate Selection**: Orders candidates by softmax-aggregated similarity scores and selects top 10 recommendations

### 6. AI-Powered Enhancement
Final refinement using advanced AI capabilities:

- **GPT-4 Reranking**: Uses GPT-4o to intelligently rerank top candidates
- **Taste Summary Generation**: Builds comprehensive user taste profile from clusters
- **Personalized Justifications**: Generates custom explanations for each recommendation
- **Cluster-Based Analysis**: Uses user preference clusters and average ratings for recommendations
- **Context Understanding**: Leverages movie titles, genres, and cluster summaries for recommendations

### 7. Response Generation
Packages final recommendations for the user:

- **Recommendation Items**: Creates structured recommendation objects with all movie details
- **Similarity Scoring**: Provides numerical similarity scores (0-1 scale) with 2 decimal precision
- **Justification Text**: Includes AI-generated explanations for why each movie was recommended
- **Metadata Enrichment**: Includes year, genres, countries, and descriptions

### Key AI Components
- **OpenAI Embeddings**: Converts movie text to high-dimensional vectors
- **GPT-4 Integration**: Provides intelligent reranking and justifications

### Traditional Algorithms
- **Fuzzy String Matching**: Uses Levenshtein distance via rapidfuzz for watched movie detection
- **Cosine Similarity**: Vector similarity calculation for movie matching
- **K-means & HDBSCAN**: Statistical clustering algorithms for user preference grouping
- **Weighted Similarity**: Uses cosine similarity with rating and recency weights for movie matching


## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Recommendation │    │   AI Services   │
│                 │    │    Manager      │    │                 │
│  • REST API     │◄──►│  • Orchestration│◄──►│  • Embeddings   │
│  • DI Container │    │  • User Profiles│    │  • Clustering   │
│  • Validation   │    │  • Filtering    │    │  • GPT-4 Rerank │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Repositories  │    │   Domain Layer  │    │  External APIs  │
│                 │    │                 │    │                 │
│  • Vector Store │    │  • Entities     │    │  • TMDB API     │
│  • User Profile │    │  • Interfaces   │    │  • OpenAI API   │
│  • Cache        │    │  • Value Objects│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Data Layer    │
│                 │
│  • PostgreSQL   │
│  • pgvector     │
│  • Redis        │
└─────────────────┘
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Łukasz Ucher** - [lukuch12@gmail.com](mailto:lukuch12@gmail.com)

---