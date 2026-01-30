0) One-time prerequisites (don’t skip)
Put your Search Admin key in a Databricks secret (recommended)

Use a secret scope (example name: kv-scope) and key name search-admin-key.

Then reference it like:

SEARCH_ADMIN_KEY = dbutils.secrets.get("kv-scope", "search-admin-key")

If you can’t use secrets yet, keep it as an env var or paste temporarily (but don’t commit it).

1) Detect embedding dimensions from your dataframe

Run this in Databricks to avoid guessing dimensions:

from pyspark.sql import functions as F


# df = your dataframe
dims = df.select(
    F.size("text_embedding").alias("text_dim"),
    F.size("image_pixel_embedding").alias("img_pixel_dim"),
    F.size("image_description_embedding").alias("img_desc_dim")
).where(
    F.col("text_dim").isNotNull() |
    F.col("img_pixel_dim").isNotNull() |
    F.col("img_desc_dim").isNotNull()
).limit(1).collect()[0].asDict()


dims

You’ll get something like:

{"text_dim": 1024, "img_pixel_dim": 1024, "img_desc_dim": 1024}

Save them:

TEXT_DIM = int(dims.get("text_dim") or 0)
IMG_PIXEL_DIM = int(dims.get("img_pixel_dim") or 0)
IMG_DESC_DIM = int(dims.get("img_desc_dim") or 0)


TEXT_DIM, IMG_PIXEL_DIM, IMG_DESC_DIM
2) Create the Azure AI Search index (REST)
Fill these placeholders

SEARCH_SERVICE_NAME (your search service)

Gov cloud endpoint: https://{service}.search.azure.us

INDEX_NAME you want (example: itsm-mm-index)

import requests, json


SEARCH_SERVICE_NAME = "itsmmulti"          # <-- your service name
INDEX_NAME = "itsm-mm-index"               # <-- pick a name
API_VERSION = "2023-11-01"                 # stable for vector search


SEARCH_ENDPOINT = f"https://{SEARCH_SERVICE_NAME}.search.azure.us"
SEARCH_ADMIN_KEY = dbutils.secrets.get("kv-scope", "search-admin-key")  # recommended


headers = {
    "Content-Type": "application/json",
    "api-key": SEARCH_ADMIN_KEY
}


# Basic sanity check
r = requests.get(f"{SEARCH_ENDPOINT}/indexes?api-version={API_VERSION}", headers=headers, timeout=60)
print(r.status_code, r.text[:500])
Index definition (matches your schema + adds vector search)

This creates three vector fields so you can do:

text-to-text retrieval (text_embedding)

text-to-image retrieval (image_description_embedding)

image-to-image retrieval (image_pixel_embedding) if you ever add image queries

index_schema = {
  "name": INDEX_NAME,
  "fields": [
    {"name": "id", "type": "Edm.String", "key": True,  "filterable": True, "sortable": True},
    {"name": "doc_id", "type": "Edm.String", "filterable": True, "sortable": True},
    {"name": "file_name", "type": "Edm.String", "filterable": True, "sortable": True},
    {"name": "page_num", "type": "Edm.Int32", "filterable": True, "sortable": True},
    {"name": "item_type", "type": "Edm.String", "filterable": True, "facetable": True},
    {"name": "content", "type": "Edm.String", "searchable": True},
    {"name": "image_url", "type": "Edm.String", "filterable": False},
    {"name": "pdf_url", "type": "Edm.String", "filterable": False},


    # Vector fields
    {
      "name": "text_embedding",
      "type": "Collection(Edm.Single)",
      "searchable": True,
      "dimensions": TEXT_DIM,
      "vectorSearchProfile": "hnsw-profile"
    },
    {
      "name": "image_pixel_embedding",
      "type": "Collection(Edm.Single)",
      "searchable": True,
      "dimensions": IMG_PIXEL_DIM,
      "vectorSearchProfile": "hnsw-profile"
    },
    {
      "name": "image_description_embedding",
      "type": "Collection(Edm.Single)",
      "searchable": True,
      "dimensions": IMG_DESC_DIM,
      "vectorSearchProfile": "hnsw-profile"
    }
  ],
  "vectorSearch": {
    "algorithms": [
      {
        "name": "hnsw",
        "kind": "hnsw",
        "hnswParameters": {"m": 8, "efConstruction": 200, "efSearch": 100, "metric": "cosine"}
      }
    ],
    "profiles": [
      {"name": "hnsw-profile", "algorithm": "hnsw"}
    ]
  }
}


# Create or update index
url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION}"
resp = requests.put(url, headers=headers, data=json.dumps(index_schema), timeout=120)
print(resp.status_code)
print(resp.text[:800])

✅ Validation rule: This must return 201 (created) or 204 (updated) without error.

3) Push your dataframe rows to the index (batch upload)

Azure Search expects documents in this format:

{"value": [{"@search.action":"upload", ...doc...}, ...]}
Convert Spark rows → Python dicts safely

Key points:

embeddings must be plain Python list[float] (not Spark arrays)

avoid null vectors (set them to None or skip field)

from pyspark.sql import functions as F


# Optional: keep only columns we index
cols = ["id","doc_id","file_name","page_num","item_type","content","text_embedding",
        "image_url","image_pixel_embedding","image_description_embedding","pdf_url"]
df2 = df.select(*cols)


def row_to_doc(r):
    d = r.asDict(recursive=True)
    doc = {"@search.action": "upload"}


    # Copy scalar fields
    for k in ["id","doc_id","file_name","page_num","item_type","content","image_url","pdf_url"]:
        if d.get(k) is not None:
            doc[k] = d[k]


    # Copy vectors only if present and correct length
    for k, dim in [
        ("text_embedding", TEXT_DIM),
        ("image_pixel_embedding", IMG_PIXEL_DIM),
        ("image_description_embedding", IMG_DESC_DIM),
    ]:
        v = d.get(k)
        if v is not None and len(v) == dim:
            # ensure float
            doc[k] = [float(x) for x in v]


    return doc
Batch uploader
import math, time


def upload_batch(docs):
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/index?api-version={API_VERSION}"
    payload = {"value": docs}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed {r.status_code}: {r.text[:800]}")
    return r.json()


BATCH_SIZE = 500  # safe default


rows = df2.toLocalIterator()  # stream rows (avoid collect on huge sets)


batch = []
total = 0


for r in rows:
    batch.append(row_to_doc(r))
    if len(batch) >= BATCH_SIZE:
        res = upload_batch(batch)
        total += len(batch)
        print(f"Uploaded: {total}")
        batch = []
        time.sleep(0.2)


if batch:
    res = upload_batch(batch)
    total += len(batch)
    print(f"Uploaded: {total}")

✅ Validation rule: You should see uploads increasing, and no exceptions.

4) Quick validation queries (must do)
A) Count documents
url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/$count?api-version={API_VERSION}"
r = requests.get(url, headers=headers, timeout=60)
print(r.status_code, r.text)
B) Keyword search sanity check
url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version={API_VERSION}"
payload = {"search": "*", "top": 5}
r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
print(r.status_code)
print(r.json()["value"][0].keys())