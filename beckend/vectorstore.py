import os
import logging
from pathlib import Path
from typing import List, Tuple

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Embeddings: OpenAI, Gemini, or Sentence-Transformers (OSS)
def _get_embeddings():
    # Check if there's a separate embeddings provider setting
    provider = os.getenv("EMBED_PROVIDER", os.getenv("PROVIDER", "openai")).lower()
    logger.info(f"Initializing embeddings with provider: {provider}")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        logger.info(f"Using OpenAI embeddings model: {model}")
        return OpenAIEmbeddings(model=model)
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        logger.info(f"Using Gemini embeddings model: {model}")
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key
        )
    else:
        # Free option
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Using HuggingFace embeddings model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)

def extract_text_from_pdfs(pdf_paths: List[str]) -> List[Tuple[str, int, str]]:
    """Return a row per PDF page as (source_name, page_number, text)."""
    from PyPDF2 import PdfReader
    results = []
    for p in pdf_paths:
        reader = PdfReader(p)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            results.append((Path(p).name, page_number, text))
    return results

def chunk_text(
    source_text_pairs: List[Tuple[str, int, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Tuple[List[Document], List[Tuple[str, int, int, str]]]:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    docs: List[Document] = []
    chunk_rows: List[Tuple[str, int, int, str]] = []
    for source, page_number, text in source_text_pairs:
        parts = splitter.split_text(text)
        for chunk_idx, chunk in enumerate(parts, start=1):
            metadata = {"source": source, "page": page_number, "chunk_id": chunk_idx}
            docs.append(Document(page_content=chunk, metadata=metadata))
            chunk_rows.append((source, page_number, chunk_idx, chunk))
    return docs, chunk_rows

def build_or_load_faiss(docs: List[Document], persist_dir: str = "vectorstore/faiss") -> FAISS:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    index_file = Path(persist_dir) / "index.faiss"
    store_file = Path(persist_dir) / "index.pkl"

    embeddings = _get_embeddings()
    if not docs:
        if index_file.exists() and store_file.exists():
            logger.info("Loading existing FAISS index")
            return FAISS.load_local(str(persist_dir), embeddings, allow_dangerous_deserialization=True)
        raise ValueError("No documents supplied and no existing FAISS index to load.")

    logger.info(f"Building FAISS index with {len(docs)} documents using cosine similarity")

    # Use cosine similarity by setting normalize_L2=True
    # This normalizes vectors before storing, making L2 distance equivalent to cosine similarity
    vectorstore = FAISS.from_documents(
        docs,
        embeddings,
        normalize_L2=True  # Enable cosine similarity
    )

    vectorstore.save_local(str(persist_dir))
    logger.info(f"FAISS index saved to {persist_dir}")
    return vectorstore
