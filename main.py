from crewai import Crew, Process
from src.agents import researcher, writer
from src.tasks import research_task, write_task
from src.vector_index import query_suppliers


def _build_chroma_context(need: str, n_results: int = 8) -> str:
    response = query_suppliers(query=need, n_results=n_results)
    docs = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    if not docs:
        return "No local supplier matches found in ChromaDB."

    lines = [f"Local ChromaDB matches for '{need}':"]
    for idx, doc in enumerate(docs, start=1):
        meta = metadatas[idx - 1] if idx - 1 < len(metadatas) else {}
        distance = distances[idx - 1] if idx - 1 < len(distances) else None
        supplier = meta.get("supplier", "unknown") if isinstance(meta, dict) else "unknown"
        specialization = meta.get("specialization", "-") if isinstance(meta, dict) else "-"
        location_parts = []
        if isinstance(meta, dict):
            if meta.get("city"):
                location_parts.append(str(meta["city"]))
            if meta.get("country"):
                location_parts.append(str(meta["country"]))
        location = ", ".join(location_parts) if location_parts else "-"
        similarity = "-"
        if isinstance(distance, (float, int)):
            similarity = f"{1 - float(distance):.3f}"

        lines.append(
            f"{idx}. Supplier: {supplier} | Specialization: {specialization} | "
            f"Location: {location} | Similarity: {similarity}\n{doc}"
        )

    return "\n\n".join(lines)

def run_system():
    # Assemble the crew
    manufacturing_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential # One after the other
    )

    # Ask the user what they need
    user_need = input("What manufacturing service do you need? (e.g. CNC or 3D Printing): ")

    # Start!
    print(f"\nSearching for {user_need}...\n")
    local_supplier_context = _build_chroma_context(user_need)
    result = manufacturing_crew.kickoff(
        inputs={
            'need': user_need,
            'local_supplier_context': local_supplier_context,
        }
    )
    
    print("\nDONE! Check the 'outputs/final_report.md' file.")

if __name__ == "__main__":
    run_system()