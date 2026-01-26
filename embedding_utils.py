# Embedding utilities for multimodal pipeline
# Uses open-source models: sentence-transformers for text, CLIP for images

from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, types as T
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache (loaded once per executor)
_text_model = None
_clip_model = None

def get_text_embedding_model():
    """Load text embedding model (cached per executor)."""
    global _text_model
    if _text_model is None:
        logger.info(f"Loading text embedding model: {TEXT_EMBEDDING_MODEL}")
        _text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        logger.info("Text model loaded successfully")
    return _text_model

def get_clip_model():
    """Load CLIP model (cached per executor)."""
    global _clip_model
    if _clip_model is None:
        logger.info(f"Loading CLIP model: {IMAGE_CLIP_MODEL}")
        _clip_model = SentenceTransformer(IMAGE_CLIP_MODEL)
        logger.info("CLIP model loaded successfully")
    return _clip_model

# ---------- Text Embedding Functions ----------

def generate_text_embedding(text: str) -> list:
    """
    Generate embedding for text using sentence-transformers.
    Returns 384-dimensional vector.
    """
    if not text or len(text.strip()) == 0:
        return [0.0] * TEXT_EMBEDDING_DIM
    
    try:
        model = get_text_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating text embedding: {e}")
        return [0.0] * TEXT_EMBEDDING_DIM

def generate_text_embeddings_batch(texts: pd.Series) -> pd.Series:
    """
    Batch process text embeddings for better performance.
    Used in mapInPandas for distributed processing.
    """
    try:
        model = get_text_embedding_model()
        # Filter out empty texts
        valid_texts = [t if t and len(t.strip()) > 0 else " " for t in texts]
        embeddings = model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
        return pd.Series([emb.tolist() for emb in embeddings])
    except Exception as e:
        logger.error(f"Error in batch text embedding: {e}")
        return pd.Series([[0.0] * TEXT_EMBEDDING_DIM] * len(texts))

# ---------- Image Embedding Functions ----------

def download_image(url: str, timeout: int = 10) -> Image.Image:
    """Download image from URL and return PIL Image."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return None

def generate_image_pixel_embedding(image_url: str) -> list:
    """
    Generate embedding for image pixels using CLIP vision encoder.
    Returns 512-dimensional vector.
    """
    if not image_url:
        return [0.0] * IMAGE_EMBEDDING_DIM
    
    try:
        # Download image
        img = download_image(image_url)
        if img is None:
            return [0.0] * IMAGE_EMBEDDING_DIM
        
        # Generate embedding with CLIP
        model = get_clip_model()
        embedding = model.encode(img, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating image pixel embedding for {image_url}: {e}")
        return [0.0] * IMAGE_EMBEDDING_DIM

def generate_image_description_embedding(image_url: str, description: str = None) -> list:
    """
    Generate embedding for image description using CLIP text encoder.
    If description is not provided, generates a generic one.
    Returns 512-dimensional vector.
    """
    if not image_url:
        return [0.0] * IMAGE_EMBEDDING_DIM
    
    try:
        # If no description provided, create a generic one based on image metadata
        if not description:
            description = f"Image extracted from PDF document"
        
        # Generate embedding with CLIP (text model)
        model = get_clip_model()
        embedding = model.encode(description, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating description embedding: {e}")
        return [0.0] * IMAGE_EMBEDDING_DIM

def generate_image_embeddings_batch(image_urls: pd.Series) -> tuple:
    """
    Batch process image embeddings for better performance.
    Returns both pixel embeddings and description embeddings.
    """
    model = get_clip_model()
    
    pixel_embeddings = []
    desc_embeddings = []
    
    for url in image_urls:
        if not url:
            pixel_embeddings.append([0.0] * IMAGE_EMBEDDING_DIM)
            desc_embeddings.append([0.0] * IMAGE_EMBEDDING_DIM)
            continue
        
        try:
            # Download and encode image (pixel embedding)
            img = download_image(url)
            if img:
                pixel_emb = model.encode(img, convert_to_numpy=True).tolist()
                pixel_embeddings.append(pixel_emb)
                
                # Description embedding (generic for now)
                desc_emb = model.encode("Image from PDF document", convert_to_numpy=True).tolist()
                desc_embeddings.append(desc_emb)
            else:
                pixel_embeddings.append([0.0] * IMAGE_EMBEDDING_DIM)
                desc_embeddings.append([0.0] * IMAGE_EMBEDDING_DIM)
        except Exception as e:
            logger.error(f"Error processing image {url}: {e}")
            pixel_embeddings.append([0.0] * IMAGE_EMBEDDING_DIM)
            desc_embeddings.append([0.0] * IMAGE_EMBEDDING_DIM)
    
    return pd.Series(pixel_embeddings), pd.Series(desc_embeddings)

# ---------- Spark UDFs ----------

# Simple UDFs (use for small datasets or testing)
generate_text_embedding_udf = F.udf(generate_text_embedding, T.ArrayType(T.FloatType()))
generate_image_pixel_embedding_udf = F.udf(generate_image_pixel_embedding, T.ArrayType(T.FloatType()))
generate_image_description_embedding_udf = F.udf(generate_image_description_embedding, T.ArrayType(T.FloatType()))

# ---------- Batch Processing Functions (Recommended for Production) ----------

def embed_text_batch_mapInPandas(iterator):
    """
    MapInPandas function for batch text embedding.
    More efficient than UDFs for large datasets.
    """
    for batch in iterator:
        if not batch.empty and 'content' in batch.columns:
            batch['text_embedding'] = generate_text_embeddings_batch(batch['content'])
        else:
            batch['text_embedding'] = [[0.0] * TEXT_EMBEDDING_DIM] * len(batch)
        yield batch

def embed_images_batch_mapInPandas(iterator):
    """
    MapInPandas function for batch image embedding.
    Generates both pixel and description embeddings.
    """
    for batch in iterator:
        if not batch.empty and 'image_url' in batch.columns:
            pixel_embs, desc_embs = generate_image_embeddings_batch(batch['image_url'])
            batch['image_pixel_embedding'] = pixel_embs
            batch['image_description_embedding'] = desc_embs
        else:
            batch['image_pixel_embedding'] = [[0.0] * IMAGE_EMBEDDING_DIM] * len(batch)
            batch['image_description_embedding'] = [[0.0] * IMAGE_EMBEDDING_DIM] * len(batch)
        yield batch

# ---------- Testing Functions ----------

def test_embeddings():
    """Test embedding generation with sample data."""
    print("=" * 60)
    print("Testing Embedding Models")
    print("=" * 60)
    
    # Test text embedding
    print("\n1. Testing text embedding...")
    sample_text = "This is a test document for ITSM support system."
    text_emb = generate_text_embedding(sample_text)
    print(f"   Input: {sample_text}")
    print(f"   Output dimension: {len(text_emb)}")
    print(f"   First 5 values: {text_emb[:5]}")
    
    # Test image embedding (requires valid image URL)
    print("\n2. Testing image pixel embedding...")
    print("   Note: Need valid image URL to test")
    print(f"   Expected dimension: {IMAGE_EMBEDDING_DIM}")
    
    print("\n3. Testing image description embedding...")
    desc_emb = generate_image_description_embedding(None, "Screenshot showing login page")
    print(f"   Input: 'Screenshot showing login page'")
    print(f"   Output dimension: {len(desc_emb)}")
    print(f"   First 5 values: {desc_emb[:5]}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    # For notebook testing
    test_embeddings()
