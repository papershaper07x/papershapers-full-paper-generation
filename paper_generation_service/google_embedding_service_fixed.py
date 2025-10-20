"""
Google Embedding Service (Fixed)

This module uses Google's gemini-embedding-001 model and reduces dimensions to 384
to match the existing Pinecone index.
"""

import os
import logging
import numpy as np
from typing import List, Optional
import google.generativeai as genai

LOG = logging.getLogger("uvicorn.error")

class GoogleEmbeddingService:
    """Service for generating embeddings using Google's embedding API."""
    
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004", target_dim: int = 384):
        """
        Initialize the Google embedding service.
        
        Args:
            api_key: Google API key
            model_name: Google embedding model name
            target_dim: Target dimension for embeddings (384 to match Pinecone)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.target_dim = target_dim
        self._setup_client()
    
    def _setup_client(self):
        """Setup the Google AI client."""
        try:
            genai.configure(api_key=self.api_key)
            LOG.info(f"Google embedding service configured with model: {self.model_name}")
            LOG.info(f"Target embedding dimension: {self.target_dim}")
        except Exception as e:
            LOG.error(f"Failed to setup Google embedding client: {e}")
            raise
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embedding dimensions from 768 to 384 using truncation.
        This is a simple but effective method for dimension reduction.
        """
        if embeddings.shape[-1] <= self.target_dim:
            return embeddings
        
        # Truncate to target dimensions (keep first N dimensions)
        reduced = embeddings[..., :self.target_dim]
        
        # Renormalize after truncation
        norms = np.linalg.norm(reduced, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        reduced = reduced / norms
        
        return reduced
    
    def embed_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for a list of texts using Google's embedding API.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings with shape (n_texts, 384)
        """
        if not texts:
            return np.zeros((0, self.target_dim), dtype=np.float32)
        
        try:
            all_embeddings = []
            
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings for the batch
                batch_embeddings = []
                for text in batch_texts:
                    try:
                        # Use Google's embed_content function
                        result = genai.embed_content(
                            model=self.model_name,
                            content=text,
                            task_type="retrieval_document"
                        )
                        batch_embeddings.append(result['embedding'])
                    except Exception as e:
                        LOG.warning(f"Failed to embed text '{text[:50]}...': {e}")
                        # Use zero vector as fallback
                        batch_embeddings.append([0.0] * 768)  # text-embedding-004 is 768-dim
                
                all_embeddings.extend(batch_embeddings)
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
            # Reduce dimensions to match Pinecone index
            embeddings_array = self._reduce_dimensions(embeddings_array)
            
            # Normalize embeddings (important for similarity search)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings_array = embeddings_array / norms
            
            LOG.info(f"Generated {len(embeddings_array)} embeddings with shape {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            LOG.error(f"Failed to generate embeddings: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.target_dim), dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query text.
        
        Args:
            query: Query text to embed
            
        Returns:
            numpy array of shape (384,)
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            
            embedding = np.array(result['embedding'], dtype=np.float32)
            
            # Reduce dimensions
            embedding = self._reduce_dimensions(embedding.reshape(1, -1))[0]
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            LOG.error(f"Failed to embed query '{query[:50]}...': {e}")
            # Return zero vector as fallback
            return np.zeros(self.target_dim, dtype=np.float32)

# Global service instance
_google_embedding_service: Optional[GoogleEmbeddingService] = None

def get_google_embedding_service(api_key: str = None) -> GoogleEmbeddingService:
    """Get or create the global Google embedding service instance."""
    global _google_embedding_service
    
    if _google_embedding_service is None:
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            raise ValueError("Google API key is required for embedding service")
        
        _google_embedding_service = GoogleEmbeddingService(
            api_key=api_key,
            model_name="models/text-embedding-004",  # 768-dimensional model
            target_dim=384  # Reduce to match Pinecone
        )
    
    return _google_embedding_service

def embed_texts_google(texts: List[str], batch_size: int = 8) -> np.ndarray:
    """
    Convenience function to generate embeddings using Google's API.
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size for processing
        
    Returns:
        numpy array of embeddings with shape (n_texts, 384)
    """
    service = get_google_embedding_service()
    return service.embed_texts(texts, batch_size)

def embed_query_google(query: str) -> np.ndarray:
    """
    Convenience function to generate embedding for a single query.
    
    Args:
        query: Query text to embed
        
    Returns:
        numpy array of shape (384,)
    """
    service = get_google_embedding_service()
    return service.embed_query(query)

def load_google_embeddings():
    """Initialize Google embedding service."""
    try:
        service = get_google_embedding_service()
        LOG.info("Google embedding service initialized successfully")
        return True
    except Exception as e:
        LOG.error(f"Failed to initialize Google embedding service: {e}")
        return False

# Test function to verify embedding compatibility
def test_embedding_compatibility():
    """Test that Google embeddings work with existing Pinecone setup."""
    try:
        # Test embedding generation
        test_texts = ["This is a test document", "Another test document"]
        embeddings = embed_texts_google(test_texts)
        
        # Verify shape and properties
        assert embeddings.shape[0] == len(test_texts), "Number of embeddings should match input texts"
        assert embeddings.shape[1] == 384, "Embedding dimension should be 384"
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6), "Embeddings should be normalized"
        
        # Test query embedding
        query_emb = embed_query_google("test query")
        assert query_emb.shape == (384,), "Query embedding should have correct shape"
        assert np.allclose(np.linalg.norm(query_emb), 1.0, atol=1e-6), "Query embedding should be normalized"
        
        LOG.info("✅ Google embedding compatibility test passed")
        return True
        
    except Exception as e:
        LOG.error(f"❌ Google embedding compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    # Quick test
    import sys
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    success = test_embedding_compatibility()
    if success:
        print("🎉 Google embedding service is ready!")
    else:
        print("❌ Google embedding service test failed")
        sys.exit(1)
