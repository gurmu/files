from pyspark.sql import functions as F, types as T
import re, hashlib

CATALOG = "hive_metastore"
SCHEMA  = "itsm"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# ---- Azure Gov storage endpoints ----
ACCOUNT      = "stitsmdevz33lh8"
DFS_ENDPOINT = "dfs.core.usgovcloudapi.net"      # abfss endpoint
BLOB_ENDPOINT= "blob.core.usgovcloudapi.net"     # https endpoint

PDF_CONTAINER = "pdfitsm"        # has your ~120 PDFs
IMG_CONTAINER = "itsmimages"     # empty now; we will populate extracted images here

# ---- IMPORTANT: use your Access Key "key1" ----
# Prefer secret scope:
# STORAGE_KEY = dbutils.secrets.get(scope="YOUR_SCOPE", key="key1")

# If you temporarily paste it (not recommended), ensure .strip() at least:
STORAGE_KEY = "<PASTE_KEY1_HERE>".strip()

# This is the key fix for your repeated Bronze error:
spark.conf.set(f"fs.azure.account.key.{ACCOUNT}.{DFS_ENDPOINT}", STORAGE_KEY)

# ---- ABFSS roots (must end with /) ----
PDF_ROOT = f"abfss://{PDF_CONTAINER}@{ACCOUNT}.{DFS_ENDPOINT}/"
IMG_ROOT = f"abfss://{IMG_CONTAINER}@{ACCOUNT}.{DFS_ENDPOINT}/"

print("PDF_ROOT:", PDF_ROOT)
print("IMG_ROOT:", IMG_ROOT)

def abfss_to_https(abfss_path: str) -> str:
    """
    abfss://container@account.dfs.core.usgovcloudapi.net/folder/file.pdf
    -> https://account.blob.core.usgovcloudapi.net/container/folder/file.pdf
    """
    if not abfss_path:
        return None
    m = re.match(r"abfss://([^@]+)@([^.]+)\.dfs\.core\.usgovcloudapi\.net/(.+)", abfss_path)
    if not m:
        return None
    container, account, key = m.groups()
    return f"https://{account}.{BLOB_ENDPOINT}/{container}/{key}"

abfss_to_https_udf = F.udf(abfss_to_https, T.StringType())

def sha256_str(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

sha256_udf = F.udf(sha256_str, T.StringType())

# ---------- Gold Container (Final Output) ----------
GOLD_CONTAINER = "itsmgold"                        # container for final embeddings
GOLD_ROOT = f"abfss://{GOLD_CONTAINER}@{ACCOUNT}.{DFS_ENDPOINT}/"

print("GOLD_ROOT:", GOLD_ROOT)

# ---------- Embedding Configuration ----------
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
