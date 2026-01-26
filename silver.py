# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer - Data Cleaning & Enrichment
# MAGIC 
# MAGIC Cleans and enriches bronze data:
# MAGIC 1. Clean page text → `silver_pdf_pages`
# MAGIC 2. Upload images to `itsmimages` blob storage
# MAGIC 3. Generate image URLs → `silver_pdf_images`
# MAGIC 
# MAGIC **IMPORTANT:** Run config and bronze notebooks first!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from pyspark.sql import functions as F, types as T, Row
import re
import hashlib

# Configuration - Update these values from config notebook
CATALOG = "hive_metastore"
SCHEMA = "itsm"
ACCOUNT = "stitsmdevz33lh8"
DFS_ENDPOINT = "dfs.core.usgovcloudapi.net"
BLOB_ENDPOINT = "blob.core.usgovcloudapi.net"

IMG_CONTAINER = "itsmimages"
IMG_ROOT = f"abfss://{IMG_CONTAINER}@{ACCOUNT}.{DFS_ENDPOINT}/"

# Helper function
def abfss_to_https(abfss_path: str) -> str:
    """Convert ABFSS path to HTTPS URL."""
    if not abfss_path:
        return None
    m = re.match(r"^abfss://([^@]+)@([^.]+)\.dfs\.core\.usgovcloudapi\.net/(.+)$", abfss_path)
    if not m:
        return None
    container, account, key = m.groups()
    return f"https://{account}.{BLOB_ENDPOINT}/{container}/{key}"

abfss_to_https_udf = F.udf(abfss_to_https, T.StringType())

print("=" * 80)
print("SILVER LAYER: Data Cleaning & Enrichment")
print("=" * 80)
print(f"Catalog.Schema: {CATALOG}.{SCHEMA}")
print(f"Image Storage: {IMG_ROOT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Clean Page Text

# COMMAND ----------

print("\n[1/3] Cleaning page text...")

silver_pdf_pages = (
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_pdf_pages").alias("p")
      .join(
          spark.table(f"{CATALOG}.{SCHEMA}.bronze_pdf_bin").select("doc_id", "path", "file_name").alias("b"),
          on="doc_id",
          how="left"
      )
      .withColumn("pdf_url", abfss_to_https_udf(F.col("path")))
      .withColumn("page_text_clean", F.trim(F.regexp_replace(F.col("page_text_raw"), r"\s+", " ")))
      .drop("page_text_raw", "path")
      .select("doc_id", "file_name", "page_num", "page_text_clean", "page_text_len", "pdf_url")
)

# Save silver_pdf_pages table
(silver_pdf_pages.write.mode("overwrite")
 .option("mergeSchema", "true")
 .format("delta")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.silver_pdf_pages"))

page_count = silver_pdf_pages.count()
print(f"   ✓ Cleaned {page_count:,} pages")

# Display sample
display(spark.table(f"{CATALOG}.{SCHEMA}.silver_pdf_pages").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Upload Images to Blob Storage

# COMMAND ----------

print("\n[2/3] Uploading images to itsmimages container...")
print("   Using distributed file write (this may take a few minutes)...")

import pandas as pd

def write_images_batch(iterator):
    """
    Write images to blob storage using Spark's file write.
    This runs on executors without needing SparkContext.
    """
    import io
    import re
    
    for batch in iterator:
        out_rows = []
        
        for _, r in batch.iterrows():
            doc_id = r["doc_id"]
            page_num = int(r["page_num"]) if pd.notna(r["page_num"]) else 0
            image_id = r.get("image_id", "")
            kind = r.get("image_kind", "")
            name = r.get("image_name", "")
            img_bytes = r.get("image_bytes")
            
            if img_bytes is None or len(img_bytes) == 0:
                continue
            
            # Choose extension from name
            ext = "png"
            m = re.search(r"\.([a-zA-Z0-9]+)$", name or "")
            if m:
                ext = m.group(1).lower()
            
            # Determine output path
            if kind == "page_render":
                out_key = f"{doc_id}/page_render/page_{page_num:04d}.png"
            else:
                out_key = f"{doc_id}/embedded/p{page_num:04d}_{image_id}.{ext}"
            
            # Store info for tracking
            out_rows.append({
                "doc_id": doc_id,
                "page_num": page_num,
                "image_id": image_id,
                "out_path": out_key,
                "size": len(img_bytes)
            })
        
        yield pd.DataFrame(out_rows) if out_rows else pd.DataFrame()

# Create schema for tracking
track_schema = "doc_id string, page_num int, image_id string, out_path string, size int"

# Write images using DataFrame API (avoids SparkContext on workers)
bronze_img_df = spark.table(f"{CATALOG}.{SCHEMA}.bronze_pdf_images")

# Use Spark's partitionBy to write images to correct locations
# This approach writes directly without needing Hadoop FS API on executors
from pyspark.sql import functions as F

# Write each image as a separate file
image_write_df = (
    bronze_img_df
    .withColumn("file_ext", 
        F.when(F.col("image_kind") == "page_render", F.lit("png"))
         .otherwise(F.regexp_extract("image_name", r"\.([a-zA-Z0-9]+)$", 1)))
    .withColumn("file_path",
        F.when(F.col("image_kind") == "page_render",
            F.concat(F.lit(IMG_ROOT), F.col("doc_id"), F.lit("/page_render/page_"),
                     F.lpad(F.col("page_num").cast("string"), 4, "0"), F.lit(".png")))
         .otherwise(
            F.concat(F.lit(IMG_ROOT), F.col("doc_id"), F.lit("/embedded/p"),
                     F.lpad(F.col("page_num").cast("string"), 4, "0"),
                     F.lit("_"), F.col("image_id"), F.lit("."), F.col("file_ext"))))
)

# Write each image individually using binaryFile format
# Group by doc_id for efficiency
docs = bronze_img_df.select("doc_id").distinct().collect()
total_docs = len(docs)

for idx, row in enumerate(docs):
    doc_id = row["doc_id"]
    
    # Get images for this document
    doc_images = bronze_img_df.filter(F.col("doc_id") == doc_id)
    
    # Write to temporary location then copy
    for img_row in doc_images.collect():
        page_num = int(img_row["page_num"])
        kind = img_row["image_kind"]
        img_id = img_row["image_id"]
        img_bytes = img_row["image_bytes"]
        name = img_row["image_name"]
        
        if img_bytes is None:
            continue
        
        # Determine extension
        ext = "png"
        m = re.search(r"\.([a-zA-Z0-9]+)$", name or "")
        if m:
            ext = m.group(1).lower()
        
        # Build path
        if kind == "page_render":
            out_path = f"{IMG_ROOT}{doc_id}/page_render/page_{page_num:04d}.png"
        else:
            out_path = f"{IMG_ROOT}{doc_id}/embedded/p{page_num:04d}_{img_id}.{ext}"
        
        # Write using dbutils (works on driver)
        dbutils.fs.put(out_path, bytes(img_bytes).decode('latin1'), overwrite=True)
    
    if (idx + 1) % 10 == 0:
        print(f"   Progress: {idx + 1}/{total_docs} documents processed")

print(f"   ✓ Images uploaded to itsmimages container ({total_docs} documents)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Image Metadata with URLs

# COMMAND ----------

print("\n[3/3] Creating image metadata with URLs...")

silver_pdf_images = (
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_pdf_images").alias("i")
      .join(
          spark.table(f"{CATALOG}.{SCHEMA}.bronze_pdf_bin").select("doc_id", "path", "file_name").alias("b"),
          on="doc_id",
          how="left"
      )
      .withColumn("pdf_url", abfss_to_https_udf(F.col("path")))
      .withColumn(
          "image_abfss_path",
          F.when(
              F.col("image_kind") == F.lit("page_render"),
              F.concat(
                  F.lit(IMG_ROOT), F.col("doc_id"), F.lit("/page_render/page_"),
                  F.lpad(F.col("page_num").cast("string"), 4, "0"), F.lit(".png")
              )
          ).otherwise(
              F.concat(
                  F.lit(IMG_ROOT), F.col("doc_id"), F.lit("/embedded/p"),
                  F.lpad(F.col("page_num").cast("string"), 4, "0"),
                  F.lit("_"), F.col("image_id"), F.lit("."),
                  F.regexp_extract(F.col("image_name"), r"\.([A-Za-z0-9]+)$", 1)
              )
          )
      )
      .withColumn("image_url", abfss_to_https_udf(F.col("image_abfss_path")))
      # Drop bytes to save storage in silver layer
      .drop("image_bytes", "path")
      .select(
          "doc_id", "file_name", "page_num", "image_id", "image_kind", "image_name",
          "image_mime", "width", "height", "image_abfss_path", "image_url", "pdf_url"
      )
)

# Save silver_pdf_images table
(silver_pdf_images.write.mode("overwrite")
 .option("mergeSchema", "true")
 .format("delta")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.silver_pdf_images"))

image_count = silver_pdf_images.count()
print(f"   ✓ Created metadata for {image_count:,} images")

# Display sample
display(spark.table(f"{CATALOG}.{SCHEMA}.silver_pdf_images").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 80)
print("SILVER LAYER COMPLETE")
print("=" * 80)
print(f"""
Tables Created:
  • silver_pdf_pages:    {page_count:,} pages (cleaned text + URLs)
  • silver_pdf_images:   {image_count:,} images (metadata + URLs)

Images stored in: {IMG_ROOT}

Next Step: Run the Gold layer notebook for embedding generation
""")
print("=" * 80)
