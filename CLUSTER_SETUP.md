# Databricks Cluster Setup for Multimodal Embedding Pipeline

## Required Cluster Configuration

### 1. Cluster Specifications
- **Runtime**: Databricks Runtime 14.3 LTS ML or higher (includes ML libraries)
- **Python**: 3.10+
- **Node Type**: Memory-optimized (Standard_E8ds_v4 or better)
- **Workers**: 2-8 workers depending on PDF volume

### 2. Install Python Libraries

#### Option A: Using Cluster UI (Recommended for first-time setup)
1. Go to your cluster → **Libraries** tab
2. Click **Install New**
3. Select **PyPI**
4. Install these packages one by one:
   - `sentence-transformers==2.3.1`
   - `transformers==4.36.2`
   - `torch==2.1.2`
   - `ftfy==6.1.3`
   - `Pillow==10.2.0`
   - `opencv-python==4.9.0.80`
   - `openai==1.10.0` (optional)

#### Option B: Using Init Script (Automated)
Create this init script:

```bash
#!/bin/bash
pip install --upgrade pip
pip install sentence-transformers==2.3.1 transformers==4.36.2 torch==2.1.2
pip install ftfy==6.1.3 Pillow==10.2.0 opencv-python==4.9.0.80
pip install openai==1.10.0
```

Save to DBFS: `/dbfs/databricks/scripts/install_embedding_libs.sh`

Then configure cluster:
- **Advanced Options** → **Init Scripts** → Add script path

#### Option C: Using requirements.txt
1. Upload `requirements.txt` to DBFS: `/dbfs/FileStore/requirements.txt`
2. Add init script:
```bash
#!/bin/bash
pip install -r /dbfs/FileStore/requirements.txt
```

### 3. Configure Azure Storage Access

Add these Spark configurations to your cluster:

```python
# Azure Gov Cloud storage access
spark.conf.set(
    "fs.azure.account.key.stitsmdevz33lh8.dfs.core.usgovcloudapi.net",
    dbutils.secrets.get(scope="<your-secret-scope>", key="storage-account-key")
)
```

**OR** use the notebook configuration (already in your code):

```python
storage_key = dbutils.secrets.get(scope="<scope>", key="<key>")
spark.conf.set(
    f"fs.azure.account.key.{ACCOUNT}.{DFS_ENDPOINT}",
    storage_key
)
```

### 4. Model Download (First Run)

On first run, sentence-transformers will download models:
- `clip-ViT-B-32` (~600MB)
- `all-MiniLM-L6-v2` (~80MB)

These are cached on the cluster, so subsequent runs are fast.

### 5. Execution Order

Run notebooks in this order:
1. `config.py` - Configuration
2. `bronze.py` - Extract raw data
3. `silver.py` - Clean and enrich
4. `embedding_utils.py` - Load embedding models (test)
5. `gold.py` - Generate embeddings and export to itsmgold

### 6. Troubleshooting

**Issue: "No module named 'sentence_transformers'"**
- Solution: Install packages via cluster libraries, restart cluster

**Issue: "CUDA out of memory"**
- Solution: Use CPU mode or smaller batches (adjust BATCH_SIZE in embedding_utils.py)

**Issue: "Model download timeout"**
- Solution: Pre-download models to DBFS and load from there

**Issue: Image download fails**
- Solution: Check blob storage permissions and image URLs

### 7. Performance Optimization

For large PDF sets (100+ files):
- Increase worker nodes
- Use Delta table caching: `.cache()` on frequently accessed tables
- Batch embedding generation (default: 32 items per batch)
- Use GPU-enabled clusters for faster embedding generation
