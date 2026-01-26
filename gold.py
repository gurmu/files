# Gold Layer - Generate embeddings and export to itsmgold container
# Creates:
#   1. gold_pdf_text_chunks - Text chunks with embeddings
#   2. gold_pdf_image_items - Images with dual embeddings (pixel + description)
#   3. gold_pdf_multimodal_unified - Combined text + image table
#   4. Exports to itsmgold container (Parquet + JSON)

from pyspark.sql import functions as F, types as T
import pandas as pd

# Import embedding utilities
exec(open("embedding_utils.py").read())

print("=" * 80)
print("GOLD LAYER: Multimodal Embedding Generation")
print("=" * 80)

# ---------- Helper: Text Chunking ----------

def chunk_words(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks by word count."""
    if not text:
        return []
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if chunk.strip():  # Only add non-empty chunks
            out.append(chunk)
        i += max(1, (size - overlap))
    return out if out else [""]  # Return at least one empty string if no content

chunk_udf = F.udf(lambda s: chunk_words(s, CHUNK_SIZE, OVERLAP), T.ArrayType(T.StringType()))

# ========================================
# STEP 1: Create Text Chunks with Embeddings
# ========================================

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

print(f"   Generated {text_chunks_exploded.count()} text chunks")

# Generate text embeddings using batch processing
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

print(f"   ✓ Saved {gold_pdf_text_chunks.count()} text chunks with embeddings")

# ========================================
# STEP 2: Create Image Items with Dual Embeddings
# ========================================

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

print(f"   Found {image_items_prep.count()} images")

# Generate dual image embeddings
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

print(f"   ✓ Saved {gold_pdf_image_items.count()} image items with dual embeddings")

# ========================================
# STEP 3: Create Unified Multimodal Table
# ========================================

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
print(f"   ✓ Saved {unified_count} total items (text + images)")

# ========================================
# STEP 4: Export to itsmgold Container (Parquet)
# ========================================

print("\n" + "=" * 80)
print("EXPORTING TO ITSMGOLD CONTAINER")
print("=" * 80)

parquet_path = f"{GOLD_ROOT}multimodal_embeddings_parquet/"
print(f"\n[1/2] Exporting to Parquet: {parquet_path}")

(gold_pdf_multimodal_unified.write.mode("overwrite")
 .format("parquet")
 .option("compression", "snappy")
 .save(parquet_path))

print("   ✓ Parquet export complete")

# ========================================
# STEP 5: Export to itsmgold Container (JSON)
# ========================================

json_path = f"{GOLD_ROOT}multimodal_embeddings_json/"
print(f"\n[2/2] Exporting to JSON: {json_path}")

(gold_pdf_multimodal_unified.write.mode("overwrite")
 .format("json")
 .save(json_path))

print("   ✓ JSON export complete")

# ========================================
# Summary Statistics
# ========================================

print("\n" + "=" * 80)
print("GOLD LAYER COMPLETE - SUMMARY")
print("=" * 80)

text_count = gold_pdf_text_chunks.count()
image_count = gold_pdf_image_items.count()

print(f"""
Gold Tables Created:
  • gold_pdf_text_chunks:           {text_count:,} items
  • gold_pdf_image_items:           {image_count:,} items
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

# Display sample data
print("\nSample from unified table:")
display(gold_pdf_multimodal_unified.limit(10))
