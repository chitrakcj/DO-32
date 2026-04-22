import os
from pathlib import Path
from crewai import Agent, LLM
from tools import ChromaSupplierSearchTool, WebSearchTool
from dotenv import load_dotenv

# Always load .env from the project root, regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise RuntimeError(
        "Missing Gemini API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env file."
    )


def _build_llm(model_name: str) -> LLM:
    normalized = model_name if model_name.startswith("gemini/") else f"gemini/{model_name}"
    return LLM(model=normalized, api_key=google_api_key)


def _build_agents(model_name: str) -> tuple[Agent, Agent]:
    built_llm = _build_llm(model_name)
    built_researcher = Agent(
        role='Supplier Sourcing Agent',
        goal='Query our local ChromaDB supplier vector database for {need} first. If not enough data, check the web.',
        backstory='You are a meticulous sourcing agent who trusts internal vectorized records first.',
        tools=[ChromaSupplierSearchTool(), WebSearchTool()],
        llm=built_llm,
        verbose=False
    )

    built_writer = Agent(
        role='Technical Writer',
        goal='Write a professional supplier report.',
        backstory='You turn raw lists into beautiful business reports.',
        llm=built_llm,
        verbose=False
    )
    return built_researcher, built_writer

# Setup the Free Gemini AI
gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
llm = _build_llm(gemini_model)
researcher, writer = _build_agents(gemini_model)


def get_agents_for_model(model_name: str) -> tuple[Agent, Agent]:
    return _build_agents(model_name)


def set_llm_model(model_name: str) -> None:
    """Switch global default agents to a new Gemini model at runtime."""
    global llm, gemini_model, researcher, writer
    gemini_model = model_name
    llm = _build_llm(model_name)
    researcher, writer = _build_agents(model_name)


def get_llm_model() -> str:
    return gemini_model