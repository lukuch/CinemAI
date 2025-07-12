# CinemAI - Advanced Movie Recommendation System

## Proof of Concept Overview

CinemAI is a sophisticated movie and TV show recommendation system that leverages cutting-edge AI technologies to provide personalized recommendations. This proof of concept demonstrates a FastAPI backend architecture using embeddings, clustering, and GPT-4 for intelligent movie suggestions.

**Current Status**: Backend API implementation with comprehensive AI pipeline  
**Future Plans**: Frontend development and recommendation system enhancements

## Implementation Status

### **Currently Implemented**
- **FastAPI Backend**: Complete REST API with dependency injection
- **AI Pipeline**: Full recommendation engine with embeddings, clustering, and GPT-4
- **Database Layer**: PostgreSQL with pgvector for vector similarity search
- **Caching System**: Redis for performance optimization
- **External APIs**: TMDB integration for movie data
- **Demo Data**: Rich dataset with 65 diverse movies for testing

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

### 1. User Profile Creation & Retrieval
The system first attempts to retrieve an existing user profile from the database:

- **Profile Lookup**: Queries PostgreSQL with pgvector for existing user profile
- **Demo Data Fallback**: If no profile exists, loads rich demo dataset with 50+ movies
- **High-Rated Filtering**: Focuses on movies with ratings > 4 for quality preferences
- **Text Embedding Generation**: Creates semantic embeddings from movie titles, descriptions, genres, and countries
- **Clustering Analysis**: Uses adaptive clustering based on dataset size:
  - **< 100 movies**: Single cluster (weighted average)
  - **100-500 movies**: K-means with silhouette score optimization (2-10 clusters)
  - **> 500 movies**: HDBSCAN with minimum cluster size of 10
- **Profile Storage**: Saves user profile with clusters to PostgreSQL for future use

### 2. Movie Candidate Generation
Fetches potential movie recommendations from external sources:

- **TMDB API Integration**: Retrieves movies based on user filters (genres, countries, years)
- **Filter Application**: Applies dynamic filters for genres, countries, year ranges
- **Watched Movie Exclusion**: Removes movies the user has already seen
- **Batch Processing**: Efficiently handles large candidate sets

### 3. Content Processing & Embedding
Prepares candidate movies for similarity analysis:

- **Text Representation**: Creates comprehensive text descriptions combining title, description, genres, and countries
- **Embedding Generation**: Uses OpenAI embeddings to convert text to high-dimensional vectors
- **Vector Assignment**: Attaches embeddings to movie objects for similarity calculations

### 4. Recommendation Engine
Core algorithm that matches user preferences with candidates:

- **Similarity Calculation**: Uses vector operations to compute similarity scores between user clusters and candidate movies (cosine similarity)
- **Clustering-Based Matching**: Leverages user's preference clusters to find similar movie groups
- **Ranking Algorithm**: Orders candidates by similarity scores and user preference patterns
- **Top Candidate Selection**: Identifies the most promising recommendations

### 5. AI-Powered Enhancement
Final refinement using advanced AI capabilities:

- **GPT-4 Reranking**: Uses GPT-4o to intelligently rerank top candidates
- **Personalized Justifications**: Generates custom explanations for each recommendation
- **Cluster-Based Analysis**: Uses user preference clusters and average ratings for recommendations
- **Basic Context Understanding**: Leverages movie titles, genres, and cluster summaries for recommendations

### 6. Response Generation
Packages final recommendations for the user:

- **Recommendation Items**: Creates structured recommendation objects with all movie details
- **Similarity Scoring**: Provides numerical similarity scores (0-1 scale)
- **Justification Text**: Includes AI-generated explanations for why each movie was recommended
- **Metadata Enrichment**: Includes year, genres, countries, and descriptions

### Key AI Components
- **OpenAI Embeddings**: Converts movie text to 3072-dimensional vectors
- **Adaptive Clustering**: Uses K-means (small datasets) or HDBSCAN (large datasets) based on data size
- **Vector Similarity**: Uses cosine similarity for movie matching
- **GPT-4 Integration**: Provides intelligent reranking and justifications


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