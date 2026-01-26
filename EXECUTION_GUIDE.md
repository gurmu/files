# Multimodal Embedding Pipeline - Databricks Execution Guide

## Overview
This pipeline creates multimodal embeddings from PDFs following a **medallion architecture** (Bronze → Silver → Gold) entirely within Databricks notebooks.

---

## 📋 Prerequisites

### 1. Databricks Cluster Setup
- **Runtime**: Databricks Runtime 14.3 LTS ML or higher
- **Python**: 3.10+
- **Node Type**: Memory-optimized recommended (Standard_E8ds_v4 or better)
- **Workers**: 2-8 workers depending on PDF volume

### 2. Required Python Packages

Install via **Cluster → Libraries → Install New → PyPI**:
```
sentence-transformers==2.3.1
transformers==4.36.2
torch==2.1.2
ftfy==6.1.3
Pillow==10.2.0
```

### 3. Azure Storage Authentication

Add to your first notebook cell or cluster config:
```python
# Get storage key from Databricks secrets
storage_key = dbutils.secrets.get(scope="<your-scope>", key="<your-key>")

# Configure access to Azure Gov Cloud storage
spark.conf.set(
    "fs.azure.account.key.stitsmdevz33lh8.dfs.core.usgovcloudapi.net",
    storage_key
)
```

---

## 🚀 Execution Steps

### Step 1: Run Config Notebook
```python
%run ./config
```
**What it does:**
- Defines catalog/schema (`hive_metastore.itsm`)
- Sets up container paths (`pdfitsm`, `itsmimages`, `itsmgold`)
- Configures embedding models
- Defines helper functions

**Expected time:** < 1 minute

---

### Step 2: Run Bronze Layer
```python
%run ./bronze
```
**What it does:**
- Loads PDF binaries from `pdfitsm` container
- Extracts text from each page using PyMuPDF
- Extracts embedded images and page renders
- Creates Delta tables:
  - `bronze_pdf_bin` - Raw PDFs
  - `bronze_pdf_pages` - Page text
  - `bronze_pdf_images` - Images with bytes

**Expected time:** 5-10 minutes for ~120 PDFs

**Output example:**
```
Found 120 PDF files
Extracted 1,543 pages
Extracted 3,891 images
```

---

### Step 3: Run Silver Layer
```python
%run ./silver
```
**What it does:**
- Cleans page text (removes extra whitespace)
- Uploads all images to `itsmimages` blob container
- Generates HTTPS URLs for images
- Creates Delta tables:
  - `silver_pdf_pages` - Cleaned text + URLs
  - `silver_pdf_images` - Image metadata + URLs

**Expected time:** 10-15 minutes

**Output example:**
```
Cleaned 1,543 pages
Uploaded images to itsmimages container
Created metadata for 3,891 images
```

---

### Step 4: Run Gold Layer
```python
%run ./gold
```
**What it does:**
1. Chunks text (1200 words, 150 overlap)
2. Generates text embeddings (384-dim) using `all-MiniLM-L6-v2`
3. Downloads images and generates pixel embeddings (512-dim) using CLIP
4. Generates description embeddings (512-dim) using CLIP
5. Combines into unified multimodal table
6. Exports to `itsmgold` container (Parquet + JSON)

**Expected time:** 
- First run: 40-60 minutes (includes model download ~600MB)
- Subsequent runs: 30-45 minutes

**Output example:**
```
Generated 4,621 text chunks
Saved 4,621 text chunks with embeddings
Saved 3,891 image items with dual embeddings
Saved 8,512 total items (text + images)

Exports to itsmgold container:
  ✓ Parquet: itsmgold/multimodal_embeddings_parquet/
  ✓ JSON:    itsmgold/multimodal_embeddings_json/
```

---

## 📊 Output Data Structure

### Delta Tables
All tables in `hive_metastore.itsm`:

| Table | Rows | Columns | Key Fields |
|-------|------|---------|------------|
| `gold_pdf_text_chunks` | ~4,600 | 11 | id, content, text_embedding |
| `gold_pdf_image_items` | ~3,900 | 15 | id, image_url, image_pixel_embedding, image_description_embedding |
| `gold_pdf_multimodal_unified` | ~8,500 | 11 | id, item_type, embeddings |

### Blob Storage
```
itsmgold/
├── multimodal_embeddings_parquet/   # Optimized for bulk import
│   ├── _SUCCESS
│   └── part-*.snappy.parquet
└── multimodal_embeddings_json/      # For REST APIs
    ├── _SUCCESS
    └── part-*.json
```

---

## ✅ Verification Commands

### Check Tables in Databricks
```python
# View gold unified table
display(spark.table("hive_metastore.itsm.gold_pdf_multimodal_unified").limit(10))

# Count items
total = spark.table("hive_metastore.itsm.gold_pdf_multimodal_unified").count()
print(f"Total items: {total:,}")

# Check text vs images
df = spark.table("hive_metastore.itsm.gold_pdf_multimodal_unified")
df.groupBy("item_type").count().show()
```

### Check itsmgold Container
```python
# List Parquet files
dbutils.fs.ls("abfss://itsmgold@stitsmdevz33lh8.dfs.core.usgovcloudapi.net/multimodal_embeddings_parquet/")

# Read sample Parquet
sample = spark.read.parquet("abfss://itsmgold@stitsmdevz33lh8.dfs.core.usgovcloudapi.net/multimodal_embeddings_parquet/")
display(sample.limit(5))
```

### Verify Embedding Dimensions
```python
# Check text embedding dimension
text_sample = spark.table("hive_metastore.itsm.gold_pdf_text_chunks") \
    .select("text_embedding").first()
print(f"Text embedding dimension: {len(text_sample[0])}")  # Should be 384

# Check image embedding dimension
image_sample = spark.table("hive_metastore.itsm.gold_pdf_image_items") \
    .select("image_pixel_embedding").first()
print(f"Image pixel embedding dimension: {len(image_sample[0])}")  # Should be 512
```

---

## 🔧 Troubleshooting

### Issue: "No module named 'sentence_transformers'"
**Solution:** Install packages via Cluster Libraries, then restart cluster

### Issue: "Model download timeout"
**Solution:** First run downloads ~600MB. If timeout, increase cluster timeout or pre-download:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
SentenceTransformer('sentence-transformers/clip-ViT-B-32')
```

### Issue: "Image download failed"
**Solution:** Verify images uploaded correctly in silver layer:
```python
# Check silver_pdf_images
display(spark.table("hive_metastore.itsm.silver_pdf_images") \
    .select("image_url").limit(5))
```

### Issue: "Out of memory"
**Solution:** 
- Increase cluster memory
- Reduce batch size (modify `EMBEDDING_BATCH_SIZE` in config.py)
- Use fewer workers to concentrate memory

---

## 📈 Performance Tips

For large PDF sets (500+ files):
- Use 4-8 worker cluster
- Enable Delta caching: `spark.table("...").cache()`
- Increase `EMBEDDING_BATCH_SIZE` to 64 if memory allows
- Use GPU-enabled cluster for 2-3x speedup on embeddings

---

## 🎯 Next Steps: Azure AI Search

Once your data is in `itsmgold`:

1. **Create Azure AI Search index** with vector fields:
   - `text_embedding` (384 dimensions)
   - `image_pixel_embedding` (512 dimensions)  
   - `image_description_embedding` (512 dimensions)

2. **Import data** from `itsmgold/multimodal_embeddings_parquet/`

3. **Configure hybrid search** combining:
   - Full-text search on `content`
   - Vector search on embeddings
   - Filters on `doc_id`, `page_num`, `item_type`

4. **Build agentic RAG** that can:
   - Answer questions with text chunks
   - Show relevant images when user needs visual guidance
   - Provide step-by-step screenshots for ITSM support

---

## 📁 Notebook Files

| Notebook | Purpose |
|----------|---------|
| `config.py` | Configuration & settings |
| `bronze.py` | Extract PDFs → text + images |
| `silver.py` | Clean data → upload images |
| `gold.py` | Generate embeddings → export |

**Run in order:** config → bronze → silver → gold

---

## ✨ You're Ready!

Your medallion pipeline is set up for Databricks. Run each notebook in sequence and you'll have multimodal embeddings ready for Azure AI Search! 🚀
