# Multimodal Embedding Pipeline for Databricks

## ✅ Ready to Run on Databricks

This project creates multimodal embeddings from PDFs using a clean **medallion architecture** (Bronze → Silver → Gold) in Databricks notebooks.

---

## 🎯 What It Does

Creates an agentic RAG system for ITSM support by:
1. Extracting text and images from PDFs
2. Generating text embeddings (384-dim)
3. Generating dual image embeddings (pixel + description, 512-dim each)
4. Storing unified data in `itsmgold` container
5. Ready for Azure AI Search indexing

---

## 📁 Databricks Notebooks

| Notebook | Run Time | Purpose |
|----------|----------|---------|
| **config.py** | < 1 min | Configuration & settings |
| **bronze.py** | 5-10 min | Extract PDFs → text + images |
| **silver.py** | 10-15 min | Clean data → upload images |
| **gold.py** | 30-60 min | Generate embeddings → export |

**Total Pipeline:** ~50-90 minutes (first run includes model download)

---

## 🚀 Quick Start

### 1. Install Packages on Databricks Cluster
```
Cluster → Libraries → Install New → PyPI
Add:
  - sentence-transformers==2.3.1
  - transformers==4.36.2
  - torch==2.1.2
  - ftfy==6.1.3
  - Pillow==10.2.0
```

### 2. Configure Storage Authentication
Add to first cell of config.py:
```python
storage_key = dbutils.secrets.get(scope="<scope>", key="<key>")
spark.conf.set(
    "fs.azure.account.key.stitsmdevz33lh8.dfs.core.usgovcloudapi.net",
    storage_key
)
```

### 3. Run Notebooks in Order
```python
%run ./config   # Load configuration
%run ./bronze   # Extract PDFs
%run ./silver   # Clean & upload images
%run ./gold     # Generate embeddings
```

That's it! Your embeddings will be in the `itsmgold` container.

---

## 📊 Output

### Delta Tables (hive_metastore.itsm)
- `gold_pdf_text_chunks` - Text with 384-dim embeddings
- `gold_pdf_image_items` - Images with dual 512-dim embeddings
- `gold_pdf_multimodal_unified` - Combined table

### Blob Storage (itsmgold container)
- `multimodal_embeddings_parquet/` - Parquet format
- `multimodal_embeddings_json/` - JSON format

---

## 🔍 Key Features

✅ **Self-contained notebooks** - No external imports, uses `%run` for config  
✅ **Open-source models** - sentence-transformers + CLIP (no API costs)  
✅ **Dual image embeddings** - Pixel + description for flexibility  
✅ **All image types** - Embedded + page renders (for ITSM step-by-step guides)  
✅ **Dual formats** - Parquet + JSON exports  
✅ **Production ready** - Error handling, batch processing, logging

---

## 📖 Documentation

- **EXECUTION_GUIDE.md** - Detailed step-by-step instructions
- **CLUSTER_SETUP.md** - Cluster configuration
- **requirements.txt** - Python packages list

---

## 🎓 For ITSM Agentic RAG

Perfect for ITSM support where:
- Users ask questions → Get text answers + relevant screenshots
- Text confusing? → Show step-by-step visual guides
- Cross-modal search → Text queries find relevant images

---

## 🔧 Troubleshooting

**Packages not found?** → Install via Cluster Libraries, restart cluster  
**Model download slow?** → First run downloads ~600MB from HuggingFace  
**Out of memory?** → Increase cluster size or reduce EMBEDDING_BATCH_SIZE

See EXECUTION_GUIDE.md for detailed troubleshooting.

---

## 📈 Next Steps

1. ✅ Run the notebooks (config → bronze → silver → gold)
2. ✅ Verify data in `itsmgold` container
3. ⬜ Create Azure AI Search index with vector fields
4. ⬜ Import data from itsmgold Parquet files
5. ⬜ Build agentic RAG application

---

## ✨ Ready to Go!

All notebooks are ready to run in Databricks. Start with `config.py` and work your way through the medallion layers! 🚀
