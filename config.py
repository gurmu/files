# Databricks notebook source
# MAGIC %md
# MAGIC # Configuration - Multimodal Embedding Pipeline
# MAGIC 
# MAGIC Central configuration for all medallion layers:
# MAGIC - Azure Gov Cloud storage settings
# MAGIC - Catalog and schema definitions
# MAGIC - Embedding model configuration
# MAGIC - Helper functions

# COMMAND ----------

from pyspark.sql import functions as F, types as T
import re, hashlib

# COMMAND ----------

# MAGIC %md
# MAGIC ## Catalog & Schema Configuration

# COMMAND ----------

# UC / Hive targets
CATALOG = "hive_metastore"
SCHEMA  = "itsm"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"Using catalog: {CATALOG}")
print(f"Using schema: {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Azure Gov Storage Configuration

# COMMAND ----------

# Azure Gov Storage Account
ACCOUNT       = "stitsmdevz33lh8"
DFS_ENDPOINT  = "dfs.core.usgovcloudapi.net"
BLOB_ENDPOINT = "blob.core.usgovcloudapi.net"

# Container names
PDF_CONTAINER = "pdfitsm"        # source PDFs (~120 files)
IMG_CONTAINER = "itsmimages"     # extracted images
GOLD_CONTAINER = "itsmgold"      # final embeddings

# ABFSS roots (NOTE: end with '/')
PDF_ROOT = f"abfss://{PDF_CONTAINER}@{ACCOUNT}.{DFS_ENDPOINT}/"
IMG_ROOT = f"abfss://{IMG_CONTAINER}@{ACCOUNT}.{DFS_ENDPOINT}/"
GOLD_ROOT = f"abfss://{GOLD_CONTAINER}@{ACCOUNT}.{DFS_ENDPOINT}/"

print("PDF_ROOT:", PDF_ROOT)
print("IMG_ROOT:", IMG_ROOT)
print("GOLD_ROOT:", GOLD_ROOT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embedding Model Configuration

# COMMAND ----------

# Using open-source models (sentence-transformers)
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, good quality
IMAGE_CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"         # Unified text+image

TEXT_EMBEDDING_DIM = 384    # all-MiniLM-L6-v2 dimension
IMAGE_EMBEDDING_DIM = 512   # CLIP ViT-B/32 dimension

# Chunking parameters
CHUNK_SIZE = 1200           # words per chunk
CHUNK_OVERLAP = 150         # word overlap between chunks

# Embedding batch size (for performance)
EMBEDDING_BATCH_SIZE = 32

print(f"Text model: {TEXT_EMBEDDING_MODEL} ({TEXT_EMBEDDING_DIM}-dim)")
print(f"Image model: {IMAGE_CLIP_MODEL} ({IMAGE_EMBEDDING_DIM}-dim)")
print(f"Chunk size: {CHUNK_SIZE} words with {CHUNK_OVERLAP} overlap")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def abfss_to_https(abfss_path: str) -> str:
    """
    Convert ABFSS path to HTTPS URL.
    Example: abfss://container@account.dfs.core.usgovcloudapi.net/folder/file.pdf
          -> https://account.blob.core.usgovcloudapi.net/container/folder/file.pdf
    """
    if not abfss_path:
        return None
    m = re.match(r"^abfss://([^@]+)@([^.]+)\.dfs\.core\.usgovcloudapi\.net/(.+)$", abfss_path)
    if not m:
        return None
    container, account, key = m.groups()
    return f"https://{account}.{BLOB_ENDPOINT}/{container}/{key}"

abfss_to_https_udf = F.udf(abfss_to_https, T.StringType())

def sha256_str(s: str) -> str:
    """Generate SHA256 hash of string."""
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

sha256_udf = F.udf(sha256_str, T.StringType())

print("✓ Helper functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Configuration Complete
# MAGIC 
# MAGIC All configuration loaded. You can now run:
# MAGIC - `bronze.py` for data extraction
# MAGIC - `silver.py` for data cleaning
# MAGIC - `gold.py` for embedding generation
