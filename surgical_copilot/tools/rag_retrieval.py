"""
RAG retrieval tool: queries the PDF RAG pipeline (cholecystectomy textbooks).

Uses surgical_rag for vector search → rerank → structured chunks.
Requires the RAG index to be built: cd surgical_rag && python build_index.py
"""
import sys
from pathlib import Path
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from surgical_copilot.registry import register

# Ensure workspace root is on path so we can import surgical_rag
_WORKSPACE = Path(__file__).resolve().parent.parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

_PDF_RAG_INDEX = _WORKSPACE / "surgical_rag" / "data" / "rag_index"


def is_rag_retrieval_available() -> bool:
    """Return True if the RAG index exists (tool can run)."""
    if not _PDF_RAG_INDEX.is_dir():
        return False
    # Chroma creates chroma.sqlite3 when the index is built
    return (_PDF_RAG_INDEX / "chroma.sqlite3").exists()


class RAGRetrievalInput(BaseModel):
    """Input for RAG retrieval tool."""

    query: str = Field(
        ...,
        description="Natural language question or search query about cholecystectomy, laparoscopic surgery, anatomy, procedures, or guidelines (e.g. 'How do I obtain critical view of safety?', 'What are the steps of laparoscopic cholecystectomy?')",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve (default 8)",
    )
    filter_type: Optional[str] = Field(
        default=None,
        description="Optional filter: 'procedure', 'anatomy', 'complication', or 'guideline' to narrow results by chunk type",
    )


@register("rag_retrieval")
def make_rag_retrieval_tool():
    return RAGRetrievalTool()


class RAGRetrievalTool(BaseTool):
    """Retrieve relevant text from cholecystectomy textbooks (RAG). Use for guidelines, anatomy, procedure steps, complications."""

    name: str = "rag_retrieval"
    description: str = (
        "RAG retrieval: Searches cholecystectomy textbook knowledge base for guidelines, anatomy, procedure steps, and complications. "
        "Use when the user asks about: surgical technique, critical view of safety, anatomy (cystic duct, Calot triangle), "
        "procedure steps, complications, or best practices. "
        "REQUIRED: query (the user's question or search terms). "
        "Optional: top_k (1-20, default 8), filter_type ('procedure'|'anatomy'|'complication'|'guideline'). "
        "Returns retrieved text chunks with book, chapter, and page citations."
    )
    args_schema: Type[BaseModel] = RAGRetrievalInput  # type: ignore[assignment]

    def _run(
        self,
        query: str,
        top_k: int = 8,
        filter_type: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not is_rag_retrieval_available():
            return (
                "Error: RAG index not found. Build it with:\n"
                "  cd surgical_rag && pip install -r requirements.txt && python build_index.py\n"
                "Ensure output/chunks_with_metadata.json exists (run python run_pipeline.py first)."
            )

        try:
            from surgical_rag.rag_retrieval import rag_retrieve
        except ImportError as e:
            return (
                f"Error: Could not import surgical_rag: {e}. "
                "Ensure surgical_rag is at the workspace root (sibling of surgical_copilot)."
            )

        filters = {"type": filter_type} if filter_type else None
        out = rag_retrieve(
            query=query.strip(),
            top_k=top_k,
            filters=filters,
            use_reranker=True,
        )
        chunks = out.get("chunks", [])
        if not chunks:
            return "No relevant chunks found for this query. Try rephrasing or removing filters."

        lines = []
        for i, c in enumerate(chunks, 1):
            text = c.get("text", "").strip()
            book = c.get("book", "")
            chapter = c.get("chapter", "")
            page = c.get("page", "")
            page_end = c.get("page_end", "")
            score = c.get("score", 0)
            ref = []
            if book:
                ref.append(book)
            if chapter:
                ref.append(chapter)
            if page is not None and page != "":
                if page_end and str(page_end) != str(page):
                    ref.append(f"pp. {page}-{page_end}")
                else:
                    ref.append(f"p. {page}")
            ref_str = f" [{', '.join(ref)}]" if ref else ""
            lines.append(f"[{i}] (score={score:.2f}){ref_str}\n{text}")
        return "\n\n---\n\n".join(lines)
