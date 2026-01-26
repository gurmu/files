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

def write_images_partition(rows):
    """
    Upload images to blob storage using Hadoop FS API.
    Runs distributed on executors.
    """
    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(conf)

    for r in rows:
        doc_id = r["doc_id"]
        page_num = int(r["page_num"])
        image_id = r["image_id"]
        kind = r["image_kind"]
        name = r["image_name"]
        b = r["image_bytes"]

        if b is None:
            continue

        # Choose extension from name
        ext = "png"
        m = re.search(r"\.([a-zA-Z0-9]+)$", name or "")
        if m:
            ext = m.group(1).lower()

        # Determine output path
        if kind == "page_render":
            # Standard page naming: doc_id/page_render/page_0000.png
            out_key = f"{doc_id}/page_render/page_{page_num:04d}.png"
        else:
            # Embedded images: doc_id/embedded/p0000_imageid.ext
            out_key = f"{doc_id}/embedded/p{page_num:04d}_{image_id}.{ext}"

        out_abfss = IMG_ROOT + out_key

        path = jvm.org.apache.hadoop.fs.Path(out_abfss)
        # Overwrite if exists (idempotent)
        stream = fs.create(path, True)

        # Write bytes
        stream.write(bytearray(b))
        stream.close()

# Execute distributed write
bronze_img_df = spark.table(f"{CATALOG}.{SCHEMA}.bronze_pdf_images")
bronze_img_df.rdd.foreachPartition(write_images_partition)

print("   ✓ Images uploaded to itsmimages container")

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
