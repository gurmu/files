# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer - Data Cleaning & Enrichment
# MAGIC 
# MAGIC Cleans and enriches bronze data:
# MAGIC 1. Clean page text → `silver_pdf_pages`
# MAGIC 2. Upload images to blob storage
# MAGIC 3. Generate image URLs → `silver_pdf_images`
# MAGIC 
# MAGIC **Prerequisites:** Run `config.py` and `bronze.py` first

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Configuration

# COMMAND ----------

# Import configuration from config notebook
%run ./config

# COMMAND ----------

from pyspark.sql import functions as F, Row
import re

print("=" * 80)
print("SILVER LAYER: Data Cleaning & Enrichment")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Clean Page Text

# COMMAND ----------

print("\n[1/3] Cleaning page text...")

silver_pdf_pages = (
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_pdf_pages").alias("p")
      .join(
          spark.table(f"{CATALOG}.{SCHEMA}.bronze_pdf_bin").select("doc_id","path","file_name").alias("b"),
          on="doc_id",
          how="left"
      )
      .withColumn("pdf_url", abfss_to_https_udf(F.col("path")))
      .withColumn("page_text_clean", F.trim(F.regexp_replace(F.col("page_text_raw"), r"\s+", " ")))
      .drop("page_text_raw")
)

# Save silver_pdf_pages table
(silver_pdf_pages.write.mode("overwrite")
 .option("mergeSchema","true")
 .format("delta")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.silver_pdf_pages"))

page_count = silver_pdf_pages.count()
print(f"   ✓ Cleaned {page_count} pages")

display(spark.table(f"{CATALOG}.{SCHEMA}.silver_pdf_pages").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Upload Images to Blob Storage

# COMMAND ----------

print("\n[2/3] Uploading images to itsmimages container...")

def write_images_partition(rows):
    """
    Upload images to blob storage using Hadoop FS API.
    Runs on executors in distributed fashion.
    """
    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(conf)

    for r in rows:
        doc_id    = r["doc_id"]
        page_num  = int(r["page_num"])
        image_id  = r["image_id"]
        kind      = r["image_kind"]
        name      = r["image_name"]
        b         = r["image_bytes"]

        if b is None:
            continue

        # Choose extension from name (fallback png)
        ext = "png"
        m = re.search(r"\.([a-zA-Z0-9]+)$", name or "")
        if m:
            ext = m.group(1).lower()

        if kind == "page_render":
            # Standard page naming
            out_key = f"{doc_id}/page_render/page_{page_num:04d}.png"
        else:
            out_key = f"{doc_id}/embedded/p{page_num:04d}_{image_id}.{ext}"

        out_abfss = IMG_ROOT + out_key  # IMG_ROOT already ends with '/'

        path = jvm.org.apache.hadoop.fs.Path(out_abfss)
        # Idempotent: overwrite if exists
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
          spark.table(f"{CATALOG}.{SCHEMA}.bronze_pdf_bin").select("doc_id","path","file_name").alias("b"),
          on="doc_id",
          how="left"
      )
      .withColumn("pdf_url", abfss_to_https_udf(F.col("path")))
      .withColumn(
          "image_abfss_path",
          F.when(
              F.col("image_kind") == F.lit("page_render"),
              F.concat(F.lit(IMG_ROOT), F.col("doc_id"), F.lit("/page_render/page_"),
                       F.lpad(F.col("page_num").cast("string"), 4, "0"), F.lit(".png"))
          ).otherwise(
              F.concat(F.lit(IMG_ROOT), F.col("doc_id"), F.lit("/embedded/p"),
                       F.lpad(F.col("page_num").cast("string"), 4, "0"),
                       F.lit("_"), F.col("image_id"), F.lit("."),
                       F.regexp_extract(F.col("image_name"), r"\.([A-Za-z0-9]+)$", 1))
          )
      )
      .withColumn("image_url", abfss_to_https_udf(F.col("image_abfss_path")))
      # Drop bytes in silver to save storage
      .drop("image_bytes")
)

# Save silver_pdf_images table
(silver_pdf_images.write.mode("overwrite")
 .option("mergeSchema","true")
 .format("delta")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.silver_pdf_images"))

image_count = silver_pdf_images.count()
print(f"   ✓ Created metadata for {image_count} images")

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
  • silver_pdf_pages:    {page_count} pages (cleaned text)
  • silver_pdf_images:   {image_count} images (with URLs)

Images stored in: {IMG_ROOT}

Next: Run gold.py for embedding generation
""")
print("=" * 80)
