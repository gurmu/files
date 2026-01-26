# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer - PDF Data Extraction
# MAGIC 
# MAGIC Extracts raw data from PDFs:
# MAGIC 1. Load PDF binaries → `bronze_pdf_bin`
# MAGIC 2. Extract page text → `bronze_pdf_pages`
# MAGIC 3. Extract images (embedded + rendered) → `bronze_pdf_images`
# MAGIC 
# MAGIC **Prerequisites:** Run `config.py` first

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Configuration

# COMMAND ----------

# Import configuration from config notebook
%run ./config

# COMMAND ----------

from pyspark.sql import functions as F
import fitz  # PyMuPDF
import io
import hashlib

print("=" * 80)
print("BRONZE LAYER: PDF Data Extraction")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load PDF Binaries

# COMMAND ----------

print("\n[1/3] Loading PDF binaries...")

bronze_pdf_bin = (
    spark.read.format("binaryFile")
      .option("recursiveFileLookup", "true")
      .load(PDF_ROOT)
      .filter(F.lower(F.col("path")).endswith(".pdf"))
      .select(
          F.col("path"),
          F.regexp_extract("path", r"([^/]+)$", 1).alias("file_name"),
          F.col("modificationTime"),
          F.col("length"),
          F.col("content")
      )
      .withColumn("doc_id", F.sha2(F.col("path"), 256))
)

pdf_count = bronze_pdf_bin.count()
print(f"   Found {pdf_count} PDF files")

display(bronze_pdf_bin.select("file_name","path","length").limit(20))

# COMMAND ----------

# Save bronze_pdf_bin table
print("\n   Saving bronze_pdf_bin table...")
(bronze_pdf_bin.write.mode("overwrite")
 .option("mergeSchema", "true")
 .format("delta")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_pdf_bin"))

print(f"   ✓ Saved {pdf_count} PDFs to bronze_pdf_bin")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Extract Page Text & Images

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Extraction Function

# COMMAND ----------

def stable_id(*parts) -> str:
    """Generate stable hash ID from parts."""
    raw = "||".join([str(p) for p in parts])
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()

def extract_pdf_pages_and_images(pdf_bytes: bytes, doc_id: str, render_dpi: int = 150):
    """
    Extract text and images from PDF.
    
    Returns:
      pages:  [{doc_id,page_num,page_text_raw,page_text_len}]
      images: [{doc_id,page_num,image_id,image_kind,image_name,image_mime,image_bytes,width,height}]
    """
    pages, images = [], []
    if not pdf_bytes:
        return pages, images

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pno in range(len(doc)):
        page = doc[pno]

        # ---- Page text ----
        text = page.get_text("text") or ""
        pages.append({
            "doc_id": doc_id,
            "page_num": int(pno),
            "page_text_raw": text,
            "page_text_len": int(len(text)),
        })

        # ---- Embedded images ----
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            img_bytes = base.get("image", b"")
            ext = (base.get("ext") or "bin").lower()
            mime = "image/png" if ext == "png" else ("image/jpeg" if ext in ["jpg", "jpeg"] else "application/octet-stream")

            image_name = f"p{pno:04d}_embedded_{img_idx}.{ext}"
            image_id = stable_id(doc_id, pno, "embedded", xref, image_name, len(img_bytes))

            images.append({
                "doc_id": doc_id,
                "page_num": int(pno),
                "image_id": image_id,
                "image_kind": "embedded",
                "image_name": image_name,
                "image_mime": mime,
                "image_bytes": img_bytes,
                "width": int(base.get("width") or 0),
                "height": int(base.get("height") or 0),
            })

        # ---- Rendered full page image (PNG) ----
        pix = page.get_pixmap(dpi=render_dpi, alpha=False)
        png_bytes = pix.tobytes("png")
        image_name = f"page_{pno:04d}.png"
        image_id = stable_id(doc_id, pno, "page_render", image_name, len(png_bytes))

        images.append({
            "doc_id": doc_id,
            "page_num": int(pno),
            "image_id": image_id,
            "image_kind": "page_render",
            "image_name": image_name,
            "image_mime": "image/png",
            "image_bytes": png_bytes,
            "width": int(pix.width),
            "height": int(pix.height),
        })

    doc.close()
    return pages, images

print("✓ Extraction function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract Page Text

# COMMAND ----------

print("\n[2/3] Extracting page text...")

import pandas as pd

pdfs = spark.table(f"{CATALOG}.{SCHEMA}.bronze_pdf_bin").select("doc_id", "content")

pages_schema = "doc_id string, page_num int, page_text_raw string, page_text_len int"

def pages_map(iterator):
    for pdf_batch in iterator:
        out = []
        for _, r in pdf_batch.iterrows():
            pages, _ = extract_pdf_pages_and_images(r["content"], r["doc_id"])
            out.extend(pages)
        yield pd.DataFrame(out)

bronze_pages = pdfs.mapInPandas(pages_map, schema=pages_schema)

page_count = bronze_pages.count()
print(f"   Extracted {page_count} pages")

# Save bronze_pdf_pages table
(bronze_pages.write.mode("overwrite")
 .option("mergeSchema", "true")
 .format("delta")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_pdf_pages"))

print(f"   ✓ Saved bronze_pdf_pages")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract Images

# COMMAND ----------

print("\n[3/3] Extracting images...")

images_schema = """
doc_id string, page_num int, image_id string, image_kind string, image_name string,
image_mime string, image_bytes binary, width int, height int
"""

def images_map(iterator):
    for pdf_batch in iterator:
        out = []
        for _, r in pdf_batch.iterrows():
            _, images = extract_pdf_pages_and_images(r["content"], r["doc_id"])
            out.extend(images)
        yield pd.DataFrame(out)

bronze_images = pdfs.mapInPandas(images_map, schema=images_schema)

image_count = bronze_images.count()
print(f"   Extracted {image_count} images")

# Save bronze_pdf_images table
(bronze_images.write.mode("overwrite")
 .option("mergeSchema", "true")
 .format("delta")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_pdf_images"))

print(f"   ✓ Saved bronze_pdf_images")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 80)
print("BRONZE LAYER COMPLETE")
print("=" * 80)
print(f"""
Tables Created:
  • bronze_pdf_bin:      {pdf_count} PDFs
  • bronze_pdf_pages:    {page_count} pages
  • bronze_pdf_images:   {image_count} images

Next: Run silver.py for data cleaning
""")
print("=" * 80)
