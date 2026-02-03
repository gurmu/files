0) Notebook Setup — Config + Secrets (Databricks)

Put your secrets in Databricks Secrets (recommended)
Example scope: itsm-scope

# Databricks notebook cell


# ----------------------------
# 0. CONFIG (edit these)
# ----------------------------


# Option A: Delta table you already created (from your screenshot)
DELTA_TABLE = "hive_metastore.itsm_gold_pdf_multimodal_unified"


# Option B: Storage paths where you exported parquet/json (from your screenshot)
# Example pattern (edit to your real abfss paths):
STORAGE_PARQUET_PATH = "abfss://itsmgold@<storage-account>.dfs.core.usgovcloudapi.net/multimodal_embeddings_parquet/"
STORAGE_JSON_PATH    = "abfss://itsmgold@<storage-account>.dfs.core.usgovcloudapi.net/multimodal_embeddings_json/"


# Azure AI Search
SEARCH_ENDPOINT = dbutils.secrets.get("itsm-scope", "azure_search_endpoint")  # e.g. https://itsmmulti.search.azure.us
SEARCH_KEY      = dbutils.secrets.get("itsm-scope", "azure_search_key")
SEARCH_INDEX    = "itsm-mm-index"


# Azure OpenAI (for generating query embeddings to test vector search)
AOAI_ENDPOINT   = dbutils.secrets.get("itsm-scope", "aoai_endpoint")
AOAI_KEY        = dbutils.secrets.get("itsm-scope", "aoai_key")
AOAI_EMB_DEPLOY = dbutils.secrets.get("itsm-scope", "aoai_embedding_deployment")  # embedding deployment name


print("Config loaded.")
1) Load the Multimodal Data (Delta OR Parquet)
1A) Load from Delta (recommended)
# Databricks notebook cell
df = spark.table(DELTA_TABLE)


display(df.limit(20))
print("rows:", df.count())
print("columns:", len(df.columns))
df.printSchema()
1B) Load from Storage Parquet (if you want)
# Databricks notebook cell
df_pq = spark.read.parquet(STORAGE_PARQUET_PATH)
display(df_pq.limit(20))
print("rows:", df_pq.count())
df_pq.printSchema()
2) Storage-side Validation (Is your dataset actually multimodal?)

This quickly confirms you truly have:

rows with content

rows with image_url or pdf_url

embeddings present and consistent dimension

# Databricks notebook cell
from pyspark.sql import functions as F


base = df  # change to df_pq if you loaded parquet


stats = (base
  .select(
    F.count("*").alias("rows"),
    F.sum(F.when(F.col("content").isNotNull() & (F.length("content") > 0), 1).otherwise(0)).alias("rows_with_text"),
    F.sum(F.when(F.col("image_url").isNotNull() & (F.length("image_url") > 0), 1).otherwise(0)).alias("rows_with_image_url"),
    F.sum(F.when(F.col("pdf_url").isNotNull() & (F.length("pdf_url") > 0), 1).otherwise(0)).alias("rows_with_pdf_url"),
    F.sum(F.when(F.col("text_embedding").isNotNull(), 1).otherwise(0)).alias("rows_with_text_embedding"),
    F.sum(F.when(F.col("image_pixel_embedding").isNotNull(), 1).otherwise(0)).alias("rows_with_image_pixel_embedding"),
    F.sum(F.when(F.col("image_description_embedding").isNotNull(), 1).otherwise(0)).alias("rows_with_image_description_embedding"),
  )
)


display(stats)


# Breakdown by item_type (if used)
display(base.groupBy("item_type").count().orderBy(F.desc("count")))
Check embedding dimensions (should be constant per field)
# Databricks notebook cell
dim_check = (base
  .select(
    F.size("text_embedding").alias("text_dim"),
    F.size("image_pixel_embedding").alias("img_px_dim"),
    F.size("image_description_embedding").alias("img_desc_dim")
  )
  .groupBy("text_dim", "img_px_dim", "img_desc_dim")
  .count()
  .orderBy(F.desc("count"))
)


display(dim_check)
3) Azure AI Search Querying (Keyword + Vector Hybrid)
3A) Install / import SDK (one-time per cluster)
# Databricks notebook cell
%pip install -q azure-search-documents
dbutils.library.restartPython()
# Databricks notebook cell
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import json, requests
3B) Helper: Create embeddings for your query (Azure OpenAI)

This is the cleanest way to test your index’s vector fields.

# Databricks notebook cell
import requests


def aoai_embed_text(text: str) -> list:
    """
    Returns embedding vector for input text using Azure OpenAI embeddings deployment.
    Uses the REST endpoint so it's easy in Gov clouds too.
    """
    api_version = "2024-02-15-preview"  # if your env requires different, change here
    url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_EMB_DEPLOY}/embeddings?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": AOAI_KEY}
    payload = {"input": text}


    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


# quick test
vec = aoai_embed_text("test embedding")
print("embedding length:", len(vec))
3C) Hybrid Search against Azure AI Search

IMPORTANT: update vector_fields to match your index field names.
Commonly they are identical to your delta columns:
text_embedding, image_pixel_embedding, image_description_embedding

# Databricks notebook cell
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY)
)


VECTOR_FIELDS = {
    "text": "text_embedding",
    "img_desc": "image_description_embedding",
    "img_px": "image_pixel_embedding"
}


def search_hybrid(query_text: str, top: int = 5, use_vector: bool = True, vector_field: str = "text"):
    """
    Hybrid = keyword + vector search
    vector_field: "text" | "img_desc" | "img_px"
    """
    vqs = None
    if use_vector:
        emb = aoai_embed_text(query_text)
        vqs = [VectorizedQuery(vector=emb, k=top, fields=VECTOR_FIELDS[vector_field])]


    results = search_client.search(
        search_text=query_text,
        vector_queries=vqs,
        top=top,
        select=["id","doc_id","file_name","page_num","item_type","content","image_url","pdf_url"]
    )


    out = []
    for r in results:
        out.append({
            "id": r.get("id"),
            "doc_id": r.get("doc_id"),
            "file_name": r.get("file_name"),
            "page_num": r.get("page_num"),
            "item_type": r.get("item_type"),
            "image_url": r.get("image_url"),
            "pdf_url": r.get("pdf_url"),
            "content_snippet": (r.get("content") or "")[:250]
        })
    return out


# Example query
hits = search_hybrid("Perceptio App pdf", top=5, use_vector=True, vector_field="text")
display(spark.createDataFrame(hits))
4) “Is my index well-versed?” — Automated Recall Check (Storage vs Index)

This test does something practical:

sample N rows from your delta table that have text

use their content as a query

ask search index for top K

measure whether the same doc_id appears in the returned hits

That gives you a quick retrieval sanity score.

# Databricks notebook cell
from pyspark.sql import functions as F


def evaluate_index_recall(sample_n: int = 30, top_k: int = 5, require_images_or_pdf: bool = False):
    src = base.filter(F.col("content").isNotNull() & (F.length("content") > 30))
    if require_images_or_pdf:
        src = src.filter(
            (F.col("image_url").isNotNull() & (F.length("image_url") > 0)) |
            (F.col("pdf_url").isNotNull() & (F.length("pdf_url") > 0))
        )


    sample = (src
              .select("doc_id","file_name","page_num","item_type","content","image_url","pdf_url")
              .orderBy(F.rand())
              .limit(sample_n)
              .collect())


    total = 0
    hit = 0
    details = []


    for row in sample:
        q = row["content"][:500]  # limit query length
        doc_id = row["doc_id"]


        try:
            hits = search_hybrid(q, top=top_k, use_vector=True, vector_field="text")
            returned_doc_ids = [h["doc_id"] for h in hits if h.get("doc_id") is not None]
            ok = doc_id in returned_doc_ids
        except Exception as e:
            ok = False
            hits = []
            returned_doc_ids = []
        
        total += 1
        hit += int(ok)


        details.append({
            "expected_doc_id": doc_id,
            "expected_file": row["file_name"],
            "expected_page": row["page_num"],
            "expected_item_type": row["item_type"],
            "has_image_url": bool(row["image_url"]),
            "has_pdf_url": bool(row["pdf_url"]),
            "found_in_topk": ok,
            "returned_doc_ids": returned_doc_ids[:top_k]
        })


    recall = hit / total if total else 0.0
    return recall, details


recall, details = evaluate_index_recall(sample_n=25, top_k=5, require_images_or_pdf=True)
print("Recall@5 (multimodal rows):", recall)


display(spark.createDataFrame(details))

How to interpret this:

If Recall@5 is very low, it often means:

your index isn’t using the same embedding model as your query embedder, OR

you’re searching the wrong vector field name, OR

embeddings were not populated at indexing time, OR

your index analyzer/semantic config isn’t aligned with how text was chunked

5) Validate That the Index Returns Image/PDF Fields (Not just text)

This checks whether results include image_url and pdf_url when they should.

# Databricks notebook cell
def validate_multimodal_fields(query_text: str, top: int = 10):
    hits = search_hybrid(query_text, top=top, use_vector=True, vector_field="text")
    df_hits = spark.createDataFrame(hits)


    display(df_hits)


    # simple counts
    df_hits2 = (df_hits
      .select(
        F.count("*").alias("hits"),
        F.sum(F.when(F.col("image_url").isNotNull() & (F.length("image_url") > 0), 1).otherwise(0)).alias("hits_with_image_url"),
        F.sum(F.when(F.col("pdf_url").isNotNull() & (F.length("pdf_url") > 0), 1).otherwise(0)).alias("hits_with_pdf_url"),
      )
    )
    display(df_hits2)


validate_multimodal_fields("photo guideline", top=10)
6) (Optional) Direct REST call to Azure AI Search (Good for debugging)

If SDK behavior differs from portal, REST is the ground truth.

# Databricks notebook cell
import requests, json


def search_rest_hybrid(query_text: str, top: int = 5, vector_field: str = "text_embedding"):
    api_version = "2024-07-01"  # if your gov tenant only supports earlier, set 2023-11-01
    url = f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX}/docs/search?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": SEARCH_KEY}


    emb = aoai_embed_text(query_text)


    payload = {
        "search": query_text,
        "top": top,
        "select": "id,doc_id,file_name,page_num,item_type,content,image_url,pdf_url",
        "vectorQueries": [
            {
                "vector": emb,
                "k": top,
                "fields": vector_field
            }
        ]
    }


    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


resp = search_rest_hybrid("Perceptio App pdf", top=5, vector_field="text_embedding")
print("keys:", resp.keys())
print("returned:", len(resp.get("value", [])))
print(resp["value"][0].keys())