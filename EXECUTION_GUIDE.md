# Multimodal Embedding Pipeline - Execution Guide

## Overview
This pipeline creates multimodal embeddings from PDFs for ITSM agentic RAG support, storing final data in the `itsmgold` container for Azure AI Search indexing.

---

## Quick Start

### 1. Prerequisites
- Databricks cluster with ML runtime 14.3+ LTS
- Access to Azure Gov Cloud storage account: `stitsmdevz33lh8`
- Containers created: `pdfitsm`, `itsmimages`, `itsmgold`
- Python packages installed (see CLUSTER_SETUP.md)

### 2. Install Required Packages

**Option A: Via Cluster UI** (Recommended)
```
1. Cluster → Libraries → Install New → PyPI
2. Install these packages:
   - sentence-transformers==2.3.1
   - transformers==4.36.2
   - torch==2.1.2
   - ftfy==6.1.3
   - Pillow==10.2.0
```

**Option B: Via Init Script**
Upload `requirements.txt` to DBFS and add init script to cluster.

### 3. Configure Storage Authentication

Add to your notebook or cluster config:

```python
# Get storage key from secrets
storage_key = dbutils.secrets.get(scope="<your-scope>", key="<your-key>")

# Configure access
spark.conf.set(
    "fs.azure.account.key.stitsmdevz33lh8.dfs.core.usgovcloudapi.net",
    storage_key
)
```

### 4. Execution Order

Run notebooks in this exact order:

#### Step 1: Configuration
```python
%run ./config.py
```
- Loads all configuration settings
- Defines containers and paths
- Sets embedding model parameters

#### Step 2: Bronze Layer (Data Extraction)
```python
%run ./bronze.py
```
**Output:**
- `bronze_pdf_bin` - Raw PDF binaries (~120 files)
- `bronze_pdf_pages` - Extracted page text
- `bronze_pdf_images` - Extracted images (embedded + rendered)

**Expected time:** 5-10 minutes for 120 PDFs

#### Step 3: Silver Layer (Data Cleaning)
```python
%run ./silver.py
```
**Output:**
- `silver_pdf_pages` - Cleaned text with URLs
- `silver_pdf_images` - Image metadata with URLs
- Images written to `itsmimages` container

**Expected time:** 10-15 minutes (includes image uploads)

#### Step 4: Test Embedding Models (Optional but Recommended)
```python
%run ./embedding_utils.py
# This will run test_embeddings() and show sample outputs
```
**Verifies:**
- Models load correctly
- Embeddings have correct dimensions
- No import errors

#### Step 5: Gold Layer (Embedding Generation)
```python
%run ./gold.py
```
**Output:**
- `gold_pdf_text_chunks` - Text chunks with 384-dim embeddings
- `gold_pdf_image_items` - Images with dual 512-dim embeddings
- `gold_pdf_multimodal_unified` - Combined table
- Data exported to `itsmgold` container (Parquet + JSON)

**Expected time:** 30-60 minutes depending on:
- Number of text chunks
- Number of images
- Cluster size
- First run (downloads models): +5-10 minutes

---

## Understanding the Output

### Delta Tables (in hive_metastore.itsm)

#### 1. gold_pdf_text_chunks
| Column | Type | Description |
|--------|------|-------------|
| id | string | Unique chunk ID |
| doc_id | string | Source PDF hash |
| file_name | string | PDF filename |
| page_num | int | Page number |
| item_type | string | "text" |
| content | string | Text chunk (≤1200 words) |
| text_embedding | array<float> | 384-dim embedding |
| image_url | string | null |
| image_pixel_embedding | array<float> | null |
| image_description_embedding | array<float> | null |
| pdf_url | string | Source PDF URL |

#### 2. gold_pdf_image_items
| Column | Type | Description |
|--------|------|-------------|
| id | string | Unique image ID |
| doc_id | string | Source PDF hash |
| file_name | string | PDF filename |
| page_num | int | Page number |
| item_type | string | "image" |
| content | string | null |
| text_embedding | array<float> | null |
| image_url | string | Blob storage URL |
| image_pixel_embedding | array<float> | 512-dim CLIP embedding |
| image_description_embedding | array<float> | 512-dim CLIP embedding |
| pdf_url | string | Source PDF URL |
| image_kind | string | "embedded" or "page_render" |
| image_id | string | Unique image identifier |

#### 3. gold_pdf_multimodal_unified
Union of text chunks and image items - **this is the main table for Azure AI Search**.

### Blob Storage Exports (in itsmgold container)

#### Parquet Export
```
itsmgold/multimodal_embeddings_parquet/
├── _SUCCESS
├── part-00000-xxx.snappy.parquet
├── part-00001-xxx.snappy.parquet
└── ...
```
**Use for:** Azure AI Search bulk import, efficient querying

#### JSON Export
```
itsmgold/multimodal_embeddings_json/
├── _SUCCESS
├── part-00000-xxx.json
├── part-00001-xxx.json
└── ...
```
**Use for:** Direct inspection, REST API compatibility

---

## Verification Steps

### 1. Check Bronze Tables
```python
# Check PDF count
pdf_count = spark.table("hive_metastore.itsm.bronze_pdf_bin").count()
print(f"PDFs processed: {pdf_count}")

# Check page count
page_count = spark.table("hive_metastore.itsm.bronze_pdf_pages").count()
print(f"Pages extracted: {page_count}")

# Check image count
image_count = spark.table("hive_metastore.itsm.bronze_pdf_images").count()
print(f"Images extracted: {image_count}")
```

### 2. Check Silver Tables
```python
# Sample silver pages
display(spark.table("hive_metastore.itsm.silver_pdf_pages").limit(5))

# Sample silver images
display(spark.table("hive_metastore.itsm.silver_pdf_images").limit(5))
```

### 3. Check Gold Embeddings
```python
# Check text embeddings
text_df = spark.table("hive_metastore.itsm.gold_pdf_text_chunks")
print(f"Text chunks: {text_df.count()}")

# Verify embedding dimension
sample_text = text_df.select("text_embedding").first()
print(f"Text embedding dimension: {len(sample_text[0])}")  # Should be 384

# Check image embeddings
image_df = spark.table("hive_metastore.itsm.gold_pdf_image_items")
print(f"Image items: {image_df.count()}")

# Verify embedding dimensions
sample_img = image_df.select("image_pixel_embedding").first()
print(f"Image pixel embedding dimension: {len(sample_img[0])}")  # Should be 512
```

### 4. Check itsmgold Container
```python
# List Parquet files
parquet_files = dbutils.fs.ls(f"{GOLD_ROOT}multimodal_embeddings_parquet/")
print(f"Parquet files: {len(parquet_files)}")

# List JSON files
json_files = dbutils.fs.ls(f"{GOLD_ROOT}multimodal_embeddings_json/")
print(f"JSON files: {len(json_files)}")

# Read sample from Parquet
sample_parquet = spark.read.parquet(f"{GOLD_ROOT}multimodal_embeddings_parquet/")
display(sample_parquet.limit(10))
```

---

## Troubleshooting

### Issue: "No module named 'sentence_transformers'"
**Solution:** 
1. Install packages via cluster Libraries tab
2. Restart cluster
3. Verify installation: `!pip list | grep sentence-transformers`

### Issue: "Model download timeout"
**Solution:** 
On first run, models download from HuggingFace (~600MB). If timeout occurs:
```python
# Pre-download models
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
SentenceTransformer('sentence-transformers/clip-ViT-B-32')
```

### Issue: "Image download failed"
**Solution:**
Check that images were successfully uploaded to `itsmimages` in silver layer.
```python
# Verify image URLs are accessible
sample_url = spark.table("hive_metastore.itsm.silver_pdf_images") \
    .select("image_url").first()[0]
print(sample_url)

# Try downloading manually
import requests
resp = requests.get(sample_url)
print(resp.status_code)  # Should be 200
```

### Issue: "Out of memory during embedding"
**Solution:**
Reduce batch size in `embedding_utils.py`:
```python
EMBEDDING_BATCH_SIZE = 16  # Default is 32
```

### Issue: "Zero embeddings returned"
**Solution:**
Check that content/URLs are not null:
```python
# Check for null content
null_content = spark.table("hive_metastore.itsm.silver_pdf_pages") \
    .filter(F.col("page_text_clean").isNull())
print(f"Null content rows: {null_content.count()}")
```

---

## Performance Optimization

### For Large PDF Sets (500+ files)

1. **Increase cluster size**
   - Add more workers (4-8)
   - Use memory-optimized nodes

2. **Enable caching**
   ```python
   silver_pages = spark.table("...").cache()
   ```

3. **Partition data**
   ```python
   text_chunks_exploded.repartition(100)
   ```

4. **Use GPU cluster**
   - Faster embedding generation
   - Use GPU-enabled runtime

5. **Batch size tuning**
   ```python
   EMBEDDING_BATCH_SIZE = 64  # Increase if memory allows
   ```

---

## Next Steps: Azure AI Search Integration

Once the gold data is in `itsmgold` container:

1. **Create Azure AI Search index** with vector fields
2. **Import data** from Parquet files
3. **Configure vector profiles** for text and image embeddings
4. **Set up hybrid search** (keyword + semantic + vector)
5. **Build agentic RAG** with text + image context for ITSM support

---

## File Structure Summary

```
code/
├── config.py              # Configuration (updated ✓)
├── bronze.py              # Extract PDFs (unchanged ✓)
├── silver.py              # Clean data (unchanged ✓)
├── embedding_utils.py     # Embedding models (NEW ✓)
├── gold.py                # Generate embeddings (rewritten ✓)
├── requirements.txt       # Python packages (NEW ✓)
├── CLUSTER_SETUP.md       # Setup guide (NEW ✓)
└── EXECUTION_GUIDE.md     # This file (NEW ✓)
```

All files are ready for execution! 🚀
