from crewai import Task
from agents import researcher, writer

# Task for the Researcher
research_task = Task(
    description=(
        'Use this local ChromaDB context first and prioritize it over all other sources:\n'
        '{local_supplier_context}\n\n'
        'Find suppliers for {need}. Start with local ChromaDB results; use web search only if the local context is insufficient.'
    ),
    expected_output='A list of companies with their location and specialty.',
    agent=researcher
)

# Task for the Writer (The Hand-off happens here)
write_task = Task(
    description='Take the list from the researcher and make a Markdown report with a table.',
    expected_output='A final report in Markdown format.',
    agent=writer,
    context=[research_task], # This is the hand-off!
    output_file='outputs/final_report.md'
)