from crewai.tools import BaseTool
from ddgs import DDGS
from vector_index import query_suppliers


class ChromaSupplierSearchTool(BaseTool):
    name: str = "search_supplier_vector_database"
    description: str = "Search local supplier data semantically using the ChromaDB vector database."
    @classmethod
    def _semantic_search(cls, query: str) -> str:
        response = query_suppliers(query=query, n_results=5)

        docs = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        if not docs:
            return f"No semantic matches found for '{query}'."

        lines = [f"Top semantic matches for '{query}':"]
        for idx, doc in enumerate(docs, start=1):
            meta = metadatas[idx - 1] if idx - 1 < len(metadatas) else {}
            distance = distances[idx - 1] if idx - 1 < len(distances) else None
            supplier = meta.get("supplier", "unknown") if isinstance(meta, dict) else "unknown"
            specialization = meta.get("specialization", "") if isinstance(meta, dict) else ""
            location_parts = []
            if isinstance(meta, dict):
                if meta.get("city"):
                    location_parts.append(str(meta["city"]))
                if meta.get("country"):
                    location_parts.append(str(meta["country"]))
            location = ", ".join(location_parts) if location_parts else "-"
            score_text = "-"
            if isinstance(distance, (float, int)):
                score_text = f"{1 - float(distance):.3f}"

            lines.append(
                f"{idx}. Supplier: {supplier} | Specialization: {specialization or '-'} | "
                f"Location: {location} | Similarity: {score_text}\n{doc}"
            )

        return "\n\n".join(lines)

    @classmethod
    def _run(self, query: str) -> str:
        try:
            return self._semantic_search(query)
        except Exception as exc:
            return f"ChromaDB semantic search failed: {str(exc)}"


# Backward-compatible alias for existing imports.
class SearchExcelTool(ChromaSupplierSearchTool):
    pass


class WebSearchTool(BaseTool):
    name: str = "search_web"
    description: str = "Search the public web for supplier information using DuckDuckGo."

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=5))

            if not hits:
                return f"No web results found for '{query}'."

            lines = []
            for idx, hit in enumerate(hits, start=1):
                title = hit.get("title", "No title")
                url = hit.get("href", "")
                snippet = hit.get("body", "")
                lines.append(f"{idx}. {title}\n{url}\n{snippet}")
            return "\n\n".join(lines)
        except Exception as exc:
            return f"Web search failed: {str(exc)}"