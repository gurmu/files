# Multimodal Embedding Pipeline - Databricks Standalone Notebooks

## ✅ Ready for Databricks Copy/Paste

All notebooks are **completely standalone** - just copy/paste each file into a Databricks notebook and run in sequence.

---

## 📁 Notebook Files (Medallion Architecture)

| Notebook | Run Time | Purpose |
|----------|----------|---------|
| **config.py** | < 1 min | Configuration & storage authentication |
| **bronze.py** | 5-10 min | Extract PDFs → text + images |
| **silver.py** | 10-15 min | Clean data → upload images to blob |
| **gold.py** | 30-60 min | Generate embeddings → export to itsmgold |

**Total Pipeline:** ~50-90 minutes (first run includes model download)

---

## 🚀 Quick Start Guide

### Step 1: Install Packages on Databricks Cluster

Go to **Cluster → Libraries → Install New → PyPI** and add:
```
sentence-transformers==2.3.1
transformers==4.36.2
torch==2.1.2
ftfy==6.1.3
Pillow==10.2.0
```

**Then restart your cluster!**

---

### Step 2: Create Databricks Notebooks

1. In your Databricks workspace, create 4 new notebooks:
   - `config`
   - `bronze`
   - `silver`
   - `gold`

2. Copy/paste the content of each `.py` file into its corresponding notebook

---

### Step 3: Update Storage Key in config.py

**IMPORTANT:** In the `config` notebook, replace this line:
```python
STORAGE_KEY = "<PASTE_KEY1_HERE>".strip()
```

With your actual storage account key (Key1 from Azure portal):
```python
STORAGE_KEY = "your_actual_storage_key_here".strip()
```

---

### Step 4: Run Notebooks in Sequence

1. **config** notebook - Sets up authentication and configuration
2. **bronze** notebook - Extracts PDFs from `pdfitsm` container
3. **silver** notebook - Cleans data and uploads images to `itsmimages`
4. **gold** notebook - Generates embeddings and exports to `itsmgold`

**That's it!** Your embeddings will be in the `itsmgold` container.

---

## 📊 What You'll Get

### Delta Tables (in hive_metastore.itsm)

- `gold_pdf_text_chunks` - Text chunks with 384-dim embeddings
- `gold_pdf_image_items` - Images with dual 512-dim embeddings
- `gold_pdf_multimodal_unified` - **Main table** combining text + images

### Blob Storage (itsmgold container)

- `multimodal_embeddings_parquet/` - Optimized Parquet format
- `multimodal_embeddings_json/` - JSON format for REST APIs

---

## 🔍 Key Features

✅ **Completely standalone notebooks** - No %run imports, each file is self-contained  
✅ **Open-source models** - sentence-transformers + CLIP (no API costs)  
✅ **Dual image embeddings** - Pixel + description for maximum flexibility  
✅ **All image types** - Embedded images + full page renders  
✅ **Dual export formats** - Parquet + JSON  
✅ **Ready for copy/paste** - No configuration files needed

---

## 📖 Notebook Details

### config.py
- Sets up Azure Gov Cloud storage authentication
- Defines catalog/schema (`hive_metastore.itsm`)
- Configures embedding models
- **⚠️ UPDATE YOUR STORAGE KEY HERE**

### bronze.py
- Loads PDFs from `pdfitsm` container
- Extracts text using PyMuPDF
- Extracts embedded images and page renders (150 DPI)
- Creates 3 Delta tables

### silver.py  
- Cleans page text (removes extra whitespace)
- Uploads all images to `itsmimages` container
- Generates HTTPS URLs for each image
- Creates 2 Delta tables

### gold.py
- Chunks text (1200 words, 150 overlap)
- Generates text embeddings (384-dim)
- Downloads images and generates pixel embeddings (512-dim)
- Generates  description embeddings (512-dim)
- Creates unified table and exports to `itsmgold`

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'sentence_transformers'"
**Solution:** Install packages via Cluster Libraries, then **restart cluster**

### "Storage access denied"
**Solution:** Check your STORAGE_KEY in config notebook is correct

### "Model download timeout"
**Solution:** First run downloads ~600MB. Increase cluster timeout or wait for download to complete

### "Image download failed"
**Solution:** Verify images were uploaded in silver layer. Check network connectivity.

### "Out of memory"
**Solution:** Use larger cluster or reduce EMBEDDING_BATCH_SIZE in gold.py

---

## 📈 Expected Timeline

For ~120 PDFs:

- **Config:** < 1 minute
- **Bronze:** 5-10 minutes (PDF extraction)
- **Silver:** 10-15 minutes (image upload)
- **Gold:** 30-60 minutes (embedding generation)
  - First run: +10 minutes for model download

**Total:** ~50-90 minutes

---

## 🎯 Next Steps: Azure AI Search

Once your data is in `itsmgold`:

1. **Create index** with vector fields:
   - `text_embedding` (384 dimensions)
   - `image_pixel_embedding` (512 dimensions)
   - `image_description_embedding` (512 dimensions)

2. **Import data** from `itsmgold/multimodal_embeddings_parquet/`

3. **Configure hybrid search** combining:
   - Full-text search on `content`
   - Vector search on embeddings
   - Filters on `doc_id`, `page_num`, `item_type`

4. **Build agentic RAG** that provides:
   - Text answers from documentation
   - Relevant screenshots when users need visual guidance
   - Step-by-step visual guides for ITSM support

---

## 📂 File Structure

```
code/
├── config.py              ✅ Standalone notebook
├── bronze.py              ✅ Standalone notebook
├── silver.py              ✅ Standalone notebook
├── gold.py                ✅ Standalone notebook
├── README.md              📖 This file
├── EXECUTION_GUIDE.md     📖 Detailed guide
├── CLUSTER_SETUP.md       📖 Cluster setup
└── requirements.txt       📋 Python packages
```

**Just 4 Python files to copy/paste into Databricks notebooks!**

---

## ✨ You're Ready!

1. Install packages on your cluster
2. Create 4 Databricks notebooks
3. Copy/paste each `.py` file
4. Update storage key in config
5. Run in sequence: config → bronze → silver → gold

Your multimodal embeddings will be ready for Azure AI Search! 🚀
