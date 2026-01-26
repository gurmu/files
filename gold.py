# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer - Multimodal Embedding Generation
# MAGIC 
# MAGIC This notebook:
# MAGIC 1. Creates text chunks with embeddings (384-dim)
# MAGIC 2. Generates dual image embeddings (512-dim pixel + description)
# MAGIC 3. Combines into unified multimodal table
# MAGIC 4. Exports to `itsmgold` container (Parquet + JSON)
# MAGIC 
# MAGIC **Prerequisites:** Run `config.py`, `bronze.py`, `silver.py` first

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Config and Dependencies

# COMMAND ----------

# Import configuration from config notebook
%run ./config

# COMMAND ----------

# Standard imports
from pyspark.sql import functions as F, types as T
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 80)
print("GOLD LAYER: Multimodal Embedding Generation")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Embedding Models
# MAGIC 
# MAGIC Models will be cached on the cluster for reuse across executors.

# COMMAND ----------

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

# Test model loading
print("Testing model loading...")
try:
    get_text_embedding_model()
    get_clip_model()
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Embedding Functions

# COMMAND ----------

# Text embedding functions
def generate_text_embeddings_batch(texts: pd.Series) -> pd.Series:
    """
    Batch process text embeddings for better performance.
    Returns 384-dim embeddings.
    """
    try:
        model = get_text_embedding_model()
        # Filter out empty texts
        valid_texts = [t if t and len(str(t).strip()) > 0 else " " for t in texts]
        embeddings = model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
        return pd.Series([emb.tolist() for emb in embeddings])
    except Exception as e:
        logger.error(f"Error in batch text embedding: {e}")
        return pd.Series([[0.0] * TEXT_EMBEDDING_DIM] * len(texts))

# Image helper functions
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

def generate_image_embeddings_batch(image_urls: pd.Series) -> tuple:
    """
    Batch process image embeddings.
    Returns both pixel embeddings (512-dim) and description embeddings (512-dim).
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

print("✓ Embedding functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Text Chunking Function

# COMMAND ----------

def chunk_words(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks by word count."""
    if not text:
        return []
    words = str(text).split()
    out = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if chunk.strip():  # Only add non-empty chunks
            out.append(chunk)
        i += max(1, (size - overlap))
    return out if out else [""]  # Return at least one empty string if no content

chunk_udf = F.udf(lambda s: chunk_words(s, CHUNK_SIZE, CHUNK_OVERLAP), T.ArrayType(T.StringType()))

print("✓ Chunking function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Text Chunks

# COMMAND ----------

print("\n[1/5] Creating text chunks from silver_pdf_pages...")

# Read silver pages
silver_pages = spark.table(f"{CATALOG}.{SCHEMA}.silver_pdf_pages")

# Create chunks with metadata
text_chunks_exploded = (
    silver_pages
    .withColumn("chunks", chunk_udf(F.col("page_text_clean")))
    .withColumn("chunk", F.explode("chunks"))
    .withColumn("chunk_index", F.monotonically_increasing_id())
    .withColumn(
        "chunk_id",
        F.sha2(
            F.concat_ws("||",
                F.col("doc_id"),
                F.col("page_num").cast("string"),
                F.col("chunk_index").cast("string"),
                F.col("chunk")
            ),
            256
        )
    )
    .select(
        F.col("chunk_id").alias("id"),
        "doc_id",
        "file_name",
        "page_num",
        "chunk",
        "pdf_url"
    )
)

chunk_count = text_chunks_exploded.count()
print(f"   Generated {chunk_count:,} text chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Generate Text Embeddings

# COMMAND ----------

print("\n[2/5] Generating text embeddings (this may take a few minutes)...")

text_chunks_schema = """
    id string,
    doc_id string,
    file_name string,
    page_num int,
    chunk string,
    pdf_url string,
    text_embedding array<float>
"""

def embed_text_chunks(iterator):
    """MapInPandas function for batch text embedding."""
    for batch in iterator:
        if not batch.empty and 'chunk' in batch.columns:
            # Generate embeddings
            embeddings = generate_text_embeddings_batch(batch['chunk'])
            batch['text_embedding'] = embeddings
        else:
            batch['text_embedding'] = [[0.0] * TEXT_EMBEDDING_DIM] * len(batch)
        yield batch

text_chunks_with_embeddings = text_chunks_exploded.mapInPandas(
    embed_text_chunks,
    schema=text_chunks_schema
)

# Add item_type and prepare final schema
gold_pdf_text_chunks = (
    text_chunks_with_embeddings
    .withColumn("item_type", F.lit("text"))
    .withColumn("content", F.col("chunk"))
    .withColumn("image_url", F.lit(None).cast("string"))
    .withColumn("image_pixel_embedding", F.lit(None).cast(T.ArrayType(T.FloatType())))
    .withColumn("image_description_embedding", F.lit(None).cast(T.ArrayType(T.FloatType())))
    .select(
        "id", "doc_id", "file_name", "page_num", "item_type",
        "content", "text_embedding",
        "image_url", "image_pixel_embedding", "image_description_embedding",
        "pdf_url"
    )
)

# Save text chunks table
print("\n   Saving gold_pdf_text_chunks table...")
(gold_pdf_text_chunks.write.mode("overwrite")
 .option("mergeSchema", "true")
 .format("delta")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_pdf_text_chunks"))

text_count = gold_pdf_text_chunks.count()
print(f"   ✓ Saved {text_count:,} text chunks with embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Image Items

# COMMAND ----------

print("\n[3/5] Creating image items from silver_pdf_images...")

# Read silver images
silver_images = spark.table(f"{CATALOG}.{SCHEMA}.silver_pdf_images")

# Create image item IDs
image_items_prep = (
    silver_images
    .withColumn(
        "id",
        F.sha2(
            F.concat_ws("||",
                F.col("doc_id"),
                F.col("page_num").cast("string"),
                F.lit("image"),
                F.col("image_id")
            ),
            256
        )
    )
    .select(
        "id", "doc_id", "file_name", "page_num",
        "image_url", "image_kind", "image_id",
        "pdf_url", "width", "height", "image_mime"
    )
)

image_count = image_items_prep.count()
print(f"   Found {image_count:,} images")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generate Image Embeddings

# COMMAND ----------

print("\n[4/5] Generating image embeddings (pixel + description)...")
print("   This may take longer as it downloads and processes images...")

image_items_schema = """
    id string,
    doc_id string,
    file_name string,
    page_num int,
    image_url string,
    image_kind string,
    image_id string,
    pdf_url string,
    width int,
    height int,
    image_mime string,
    image_pixel_embedding array<float>,
    image_description_embedding array<float>
"""

def embed_images(iterator):
    """MapInPandas function for batch image embedding."""
    for batch in iterator:
        if not batch.empty and 'image_url' in batch.columns:
            # Generate both pixel and description embeddings
            pixel_embs, desc_embs = generate_image_embeddings_batch(batch['image_url'])
            batch['image_pixel_embedding'] = pixel_embs
            batch['image_description_embedding'] = desc_embs
        else:
            batch['image_pixel_embedding'] = [[0.0] * IMAGE_EMBEDDING_DIM] * len(batch)
            batch['image_description_embedding'] = [[0.0] * IMAGE_EMBEDDING_DIM] * len(batch)
        yield batch

image_items_with_embeddings = image_items_prep.mapInPandas(
    embed_images,
    schema=image_items_schema
)

# Prepare final image items schema
gold_pdf_image_items = (
    image_items_with_embeddings
    .withColumn("item_type", F.lit("image"))
    .withColumn("content", F.lit(None).cast("string"))
    .withColumn("text_embedding", F.lit(None).cast(T.ArrayType(T.FloatType())))
    .select(
        "id", "doc_id", "file_name", "page_num", "item_type",
        "content", "text_embedding",
        "image_url", "image_pixel_embedding", "image_description_embedding",
        "pdf_url", "image_kind", "image_id", "width", "height", "image_mime"
    )
)

# Save image items table
print("\n   Saving gold_pdf_image_items table...")
(gold_pdf_image_items.write.mode("overwrite")
 .option("mergeSchema", "true")
 .format("delta")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_pdf_image_items"))

final_image_count = gold_pdf_image_items.count()
print(f"   ✓ Saved {final_image_count:,} image items with dual embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Create Unified Multimodal Table

# COMMAND ----------

print("\n[5/5] Creating unified multimodal gold table...")

# Combine text and image items
gold_pdf_multimodal_unified = (
    gold_pdf_text_chunks
    .select(
        "id", "doc_id", "file_name", "page_num", "item_type",
        "content", "text_embedding",
        "image_url", "image_pixel_embedding", "image_description_embedding",
        "pdf_url"
    )
    .unionByName(
        gold_pdf_image_items.select(
            "id", "doc_id", "file_name", "page_num", "item_type",
            "content", "text_embedding",
            "image_url", "image_pixel_embedding", "image_description_embedding",
            "pdf_url"
        ),
        allowMissingColumns=True
    )
    .orderBy("doc_id", "page_num", "item_type")
)

# Save unified table
print("\n   Saving gold_pdf_multimodal_unified table...")
(gold_pdf_multimodal_unified.write.mode("overwrite")
 .option("mergeSchema", "true")
 .format("delta")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_pdf_multimodal_unified"))

unified_count = gold_pdf_multimodal_unified.count()
print(f"   ✓ Saved {unified_count:,} total items (text + images)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export to itsmgold Container

# COMMAND ----------

print("\n" + "=" * 80)
print("EXPORTING TO ITSMGOLD CONTAINER")
print("=" * 80)

# Export to Parquet
parquet_path = f"{GOLD_ROOT}multimodal_embeddings_parquet/"
print(f"\n[1/2] Exporting to Parquet: {parquet_path}")

(gold_pdf_multimodal_unified.write.mode("overwrite")
 .format("parquet")
 .option("compression", "snappy")
 .save(parquet_path))

print("   ✓ Parquet export complete")

# Export to JSON
json_path = f"{GOLD_ROOT}multimodal_embeddings_json/"
print(f"\n[2/2] Exporting to JSON: {json_path}")

(gold_pdf_multimodal_unified.write.mode("overwrite")
 .format("json")
 .save(json_path))

print("   ✓ JSON export complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary & Verification

# COMMAND ----------

print("\n" + "=" * 80)
print("GOLD LAYER COMPLETE - SUMMARY")
print("=" * 80)

print(f"""
Gold Tables Created:
  • gold_pdf_text_chunks:           {text_count:,} items
  • gold_pdf_image_items:           {final_image_count:,} items
  • gold_pdf_multimodal_unified:    {unified_count:,} items

Embedding Dimensions:
  • Text embeddings:                {TEXT_EMBEDDING_DIM}-dim
  • Image pixel embeddings:         {IMAGE_EMBEDDING_DIM}-dim
  • Image description embeddings:   {IMAGE_EMBEDDING_DIM}-dim

Exports to itsmgold container:
  ✓ Parquet: {parquet_path}
  ✓ JSON:    {json_path}

Ready for Azure AI Search indexing!
""")

print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Data Display

# COMMAND ----------

print("\nSample text chunks with embeddings:")
display(gold_pdf_text_chunks.select("id", "doc_id", "file_name", "page_num", "content", "text_embedding").limit(5))

# COMMAND ----------

print("\nSample image items with embeddings:")
display(gold_pdf_image_items.select("id", "doc_id", "file_name", "page_num", "image_url", "image_kind", "image_pixel_embedding", "image_description_embedding").limit(5))

# COMMAND ----------

print("\nSample from unified table:")
display(gold_pdf_multimodal_unified.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Gold Layer Complete
# MAGIC 
# MAGIC Your multimodal embeddings are now stored in:
# MAGIC - Delta tables in `hive_metastore.itsm`
# MAGIC - Parquet and JSON in `itsmgold` container
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC 1. Verify data in `itsmgold` container
# MAGIC 2. Create Azure AI Search index
# MAGIC 3. Import data for agentic RAG
