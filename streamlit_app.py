from __future__ import annotations

import pandas as pd
import streamlit as st

from vector_index import query_suppliers


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text else ""


def _result_rows_uncached(query: str, n_results: int = 10) -> list[dict[str, str]]:
    response = query_suppliers(query=query, n_results=n_results)
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    rows: list[dict[str, str]] = []
    for idx, meta in enumerate(metadatas):
        metadata = meta if isinstance(meta, dict) else {}
        distance = distances[idx] if idx < len(distances) else None

        material_type = _safe_text(metadata.get("material_type") or metadata.get("Category"))
        material_name = _safe_text(metadata.get("material_name") or metadata.get("Subcategory") or metadata.get("Sub_Category"))
        location = _safe_text(metadata.get("city") or metadata.get("Location"))
        cost_per_unit = _safe_text(metadata.get("cost_per_unit_usd") or metadata.get("Unit_Price"))
        reliability = _safe_text(metadata.get("reliability_score") or metadata.get("Quality_Rating"))

        similarity = "-"
        if isinstance(distance, (int, float)):
            similarity = f"{1 - float(distance):.3f}"

        rows.append(
            {
                "Supplier_Name": _safe_text(metadata.get("supplier")) or "unknown",
                "Material_Type": material_type or "-",
                "Material_Name": material_name or "-",
                "Location": location or "-",
                "Lead_Time_Days": _safe_text(metadata.get("lead_time_days")) or "-",
                "Cost_per_Unit_USD": cost_per_unit or "-",
                "Reliability_Score": reliability or "-",
                "Contact_Email": _safe_text(metadata.get("contact_email")) or "-",
                "Similarity": similarity,
            }
        )

    return rows


@st.cache_data(ttl=180, show_spinner=False)
def _result_rows(query: str, n_results: int = 10) -> list[dict[str, str]]:
    return _result_rows_uncached(query=query, n_results=n_results)


st.set_page_config(page_title="Supplier Search", page_icon="🔎", layout="wide")
st.title("Mutliagent Manufacturing System For Supplier Sourcing")
st.caption("Type in the search box to find related suppliers.")

search_text = st.text_input(
    "Search",
    placeholder="Examples: Iron, Lithium, steel..",
)

if st.button("Find Suppliers", type="primary"):
    if not search_text.strip():
        st.warning("Please enter a search query.")
    else:
        try:
            rows = _result_rows(search_text.strip(), n_results=10)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
        else:
            if not rows:
                st.info("No matching suppliers found in ChromaDB.")
            else:
                st.success(f"Found {len(rows)} supplier matches from ChromaDB.")
                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch", hide_index=True)
