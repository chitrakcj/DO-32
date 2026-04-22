from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any
import re

import pandas as pd

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:  # pragma: no cover - runtime dependency guard
    chromadb = None
    embedding_functions = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = os.getenv("SUPPLIER_CHROMA_COLLECTION", "suppliers")
FINGERPRINT_FILE = CHROMA_DIR / ".supplier_index_fingerprint"
EMBEDDING_MODEL = os.getenv("SUPPLIER_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MIN_HYBRID_SCORE = float(os.getenv("SUPPLIER_MIN_HYBRID_SCORE", "0.35"))
SEMANTIC_WEIGHT = float(os.getenv("SUPPLIER_SEMANTIC_WEIGHT", "0.75"))
KEYWORD_WEIGHT = float(os.getenv("SUPPLIER_KEYWORD_WEIGHT", "0.25"))
QUERY_CANDIDATE_MULTIPLIER = int(os.getenv("SUPPLIER_QUERY_CANDIDATE_MULTIPLIER", "2"))


def _ensure_chroma_dependencies() -> None:
    if chromadb is None or embedding_functions is None:
        raise RuntimeError(
            "ChromaDB dependencies are missing. Install with: pip install chromadb sentence-transformers"
        )


def find_supplier_file() -> Path | None:
    if not DATA_DIR.exists():
        return None
    matches = sorted(
        [
            path
            for path in DATA_DIR.glob("suppliers*")
            if path.is_file() and not path.name.startswith("~$")
        ]
    )
    return matches[0] if matches else None


def load_supplier_df(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.read_excel(file_path)
    return pd.read_excel(file_path)


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _build_document(row: pd.Series) -> tuple[str, dict[str, str]]:
    values = {column: _clean(value) for column, value in row.items()}

    preferred_fields = [
        "Supplier_Name",
        "name",
        "Company",
        "Category",
        "Subcategory",
        "Sub_Category",
        "Location",
        "Lead_Time_Days",
        "Unit_Price",
        "Cost_per_Unit_USD",
        "Quality_Rating",
        "Reliability_Score",
        "Contact_Email",
    ]

    document_parts: list[str] = []
    for field in preferred_fields:
        value = values.get(field, "")
        if value:
            document_parts.append(f"{field}: {value}")

    if not document_parts:
        document_parts = [f"{key}: {val}" for key, val in values.items() if val]

    metadata = {
        "supplier": values.get("Supplier_Name") or values.get("name") or values.get("Company") or "unknown",
        "material_type": values.get("Category") or "",
        "material_name": values.get("Subcategory") or values.get("Sub_Category") or "",
        "city": values.get("Location") or "",
        "lead_time_days": values.get("Lead_Time_Days") or "",
        "cost_per_unit_usd": values.get("Cost_per_Unit_USD") or values.get("Unit_Price") or "",
        "reliability_score": values.get("Reliability_Score") or values.get("Quality_Rating") or "",
        "contact_email": values.get("Contact_Email") or "",
    }

    return " | ".join(document_parts), metadata


def file_fingerprint(file_path: Path) -> str:
    stat = file_path.stat()
    return f"{file_path.resolve()}::{int(stat.st_mtime)}::{stat.st_size}"


def is_index_current(fingerprint: str) -> bool:
    if not FINGERPRINT_FILE.exists():
        return False
    try:
        return FINGERPRINT_FILE.read_text(encoding="utf-8").strip() == fingerprint
    except Exception:
        return False


@lru_cache(maxsize=1)
def _get_embedder():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _reset_chroma_storage() -> None:
    _is_metadata_schema_compatible.cache_clear()
    _get_collection_for_query.cache_clear()
    _get_client.cache_clear()
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _get_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return chromadb.PersistentClient(path=str(CHROMA_DIR))
    except BaseException as exc:
        message = str(exc).lower()
        if "default_tenant" in message or "tenant" in message or "panic" in message:
            _reset_chroma_storage()
            return chromadb.PersistentClient(path=str(CHROMA_DIR))
        raise


@lru_cache(maxsize=1)
def _get_collection_for_query():
    client = _get_client()
    embedder = _get_embedder()
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedder)


@lru_cache(maxsize=1)
def _is_metadata_schema_compatible() -> bool:
    try:
        collection = _get_collection_for_query()
        probe = collection.get(limit=1, include=["metadatas"])
        probe_meta = probe.get("metadatas", []) if isinstance(probe, dict) else []
        first_meta = probe_meta[0] if probe_meta and isinstance(probe_meta[0], dict) else {}
        if not first_meta:
            return True
        required_keys = {
            "material_name",
            "lead_time_days",
            "cost_per_unit_usd",
            "reliability_score",
            "contact_email",
        }
        return required_keys.issubset(set(first_meta.keys()))
    except Exception:
        return True


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 2]


def _keyword_overlap_score(query: str, candidate: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    candidate_tokens = set(_tokenize(candidate))
    if not candidate_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(candidate_tokens))
    return overlap / max(1, len(query_tokens))


def index_suppliers(force: bool = False) -> dict[str, Any]:
    _ensure_chroma_dependencies()

    supplier_file = find_supplier_file()
    if supplier_file is None:
        raise FileNotFoundError("No supplier file found in data/.")

    df = load_supplier_df(supplier_file)
    if df.empty:
        raise ValueError("Supplier data file is empty.")

    fingerprint = file_fingerprint(supplier_file)
    if not force and is_index_current(fingerprint):
        return {
            "status": "up_to_date",
            "collection": COLLECTION_NAME,
            "records": len(df),
            "source": str(supplier_file),
        }

    client = _get_client()
    embedder = _get_embedder()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    docs: list[str] = []
    metadatas: list[dict[str, str]] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        doc, metadata = _build_document(row)
        if not doc:
            continue
        ids.append(f"supplier-{idx}")
        docs.append(doc)
        metadatas.append(metadata)

    existing = collection.get(include=[])
    existing_ids = existing.get("ids", []) if isinstance(existing, dict) else []
    if existing_ids:
        collection.delete(ids=existing_ids)

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metadatas)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_FILE.write_text(fingerprint, encoding="utf-8")
    _is_metadata_schema_compatible.cache_clear()
    _get_collection_for_query.cache_clear()

    return {
        "status": "rebuilt",
        "collection": COLLECTION_NAME,
        "indexed": len(ids),
        "source": str(supplier_file),
    }


def query_suppliers(
    query: str,
    n_results: int = 5,
    subcategory: str | None = None,
    strict_subcategory: bool = False,
) -> dict[str, Any]:
    _ensure_chroma_dependencies()

    try:
        collection = _get_collection_for_query()
    except BaseException:
        # Auto-heal a missing collection by rebuilding once from source data.
        index_suppliers(force=True)
        collection = _get_collection_for_query()

    if collection.count() == 0:
        index_suppliers(force=True)
        collection = _get_collection_for_query()
        if collection.count() == 0:
            raise RuntimeError(
                "Supplier vector index is empty even after rebuild. Check data/suppliers.csv and rerun: c:/Users/Admin/Desktop/GAI08/env/Scripts/python.exe src/rebuild_vector_index.py"
            )

    # Auto-upgrade legacy indexed metadata once per process instead of on every query.
    if not _is_metadata_schema_compatible():
        index_suppliers(force=True)
        collection = _get_collection_for_query()
        _is_metadata_schema_compatible.cache_clear()
        _is_metadata_schema_compatible()

    candidate_count = max(n_results * max(1, QUERY_CANDIDATE_MULTIPLIER), 8)
    response = collection.query(
        query_texts=[query],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    docs = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    ranked: list[tuple[float, str, dict[str, Any], float]] = []
    subcategory_text = (subcategory or "").strip()
    for idx, doc in enumerate(docs):
        meta = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        distance = distances[idx] if idx < len(distances) and isinstance(distances[idx], (float, int)) else 1.0
        semantic_score = max(0.0, 1.0 - float(distance))
        keyword_source = " ".join(
            [
                str(meta.get("supplier", "")),
                str(meta.get("Location", "")),
                str(doc),
            ]
        )
        keyword_score = _keyword_overlap_score(query, keyword_source)
        subcategory_score = 0.0
        if subcategory_text:
            subcategory_source = " ".join(
                [
                    str(meta.get("material_type", "")),
                    str(meta.get("material_name", "")),
                    str(meta.get("specialization", "")),
                    str(doc),
                ]
            )
            subcategory_score = _keyword_overlap_score(subcategory_text, subcategory_source)
            if strict_subcategory and subcategory_score == 0.0:
                continue

        base_hybrid = (SEMANTIC_WEIGHT * semantic_score) + (KEYWORD_WEIGHT * keyword_score)
        if subcategory_text:
            hybrid_score = (0.70 * base_hybrid) + (0.30 * subcategory_score)
        else:
            hybrid_score = base_hybrid
        ranked.append((hybrid_score, str(doc), meta, float(distance)))

    ranked.sort(key=lambda item: item[0], reverse=True)
    filtered = [item for item in ranked if item[0] >= MIN_HYBRID_SCORE]
    selected = filtered[:n_results] if filtered else ranked[:n_results]

    final_docs = [item[1] for item in selected]
    final_metas = [item[2] for item in selected]
    final_distances = [item[3] for item in selected]

    return {
        "documents": [final_docs],
        "metadatas": [final_metas],
        "distances": [final_distances],
    }
