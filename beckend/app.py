from csv import reader
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
#from langchain.memory import ConversationBufferMemory
#from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
#from langchain_community.llms import LlamaCpp
import sqlite3, time
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .vectorstore import extract_text_from_pdfs, chunk_text, build_or_load_faiss
from .chatbot import make_conversation_chain

DB_PATH = os.getenv("SQLITE_DB_PATH", "ads_texts.db")

def get_pdf_text(pdf_docs):
    rows = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        fname = getattr(pdf, 'name', None) or os.path.basename(str(pdf))
        for idx, page in enumerate(reader.pages, start=1):
            content = page.extract_text() or ""
            rows.append((fname, idx, content))
    return rows


def get_text_chunks(rows, chunk_size=500, chunk_overlap=100):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = []
    for filename, page_no, content in rows:
        if not content.strip():
            continue
        parts = splitter.split_text(content)
        for i, part in enumerate(parts, start=1):
            chunks.append((filename, page_no, i, part))
    return chunks


def save_chunks(chunks, db_path: str = DB_PATH):
    db_path = Path(db_path)
    if db_path.parent and not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS pdf_chunks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        page INTEGER,
        chunk_id INTEGER,
        chunk_text TEXT,
        created_at INTEGER
    )""")

    now = int(time.time())
    rows = [(f, p, i, t, now) for f, p, i, t in chunks]

    if rows:
        filenames = sorted({r[0] for r in rows})
        cur.executemany(
            "DELETE FROM pdf_chunks WHERE filename = ?",
            [(name,) for name in filenames]
        )
        cur.executemany(
            "INSERT INTO pdf_chunks(filename,page,chunk_id,chunk_text,created_at) VALUES (?,?,?,?,?)",
            rows,
        )
    conn.commit()
    conn.close()

def load_documents_from_db(db_path: str) -> List[Document]:
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT filename, page, chunk_id, chunk_text FROM pdf_chunks ORDER BY filename, page, chunk_id")

    docs: List[Document] = []
    for filename, page, chunk_id, chunk_text in cur.fetchall():
        metadata = {"source": filename, "page": page, "chunk_id": chunk_id}
        docs.append(Document(page_content=chunk_text, metadata=metadata))

    conn.close()
    return docs

PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore/faiss")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
Path(UPLOAD_DIR).mkdir(exist_ok=True, parents=True)

app = FastAPI(title="DSCI-560 Q&A Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Keep a global chain instance (simple demo; production would scope per-user/session)
_chain = None

class AskBody(BaseModel):
    question: str

def _ensure_chain():
    global _chain
    if _chain is None:
        # Load existing FAISS (must exist) or create an empty placeholder error
        from langchain_community.vectorstores import FAISS
        try:
            vs = FAISS.load_local(PERSIST_DIR, embeddings=None, allow_dangerous_deserialization=True)
        except Exception:
            raise RuntimeError("Vector store not found. Ingest PDFs first via /ingest.")
        # Need embeddings object for loaded store
        from .vectorstore import _get_embeddings
        vs.embeddings = _get_embeddings()
        _chain = make_conversation_chain(vs)
    return _chain

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    paths = []
    for f in files:
        dest = Path(UPLOAD_DIR) / f.filename
        with dest.open("wb") as out:
            out.write(await f.read())
        paths.append(str(dest))

    # Extract -> Chunk -> Embed -> Save FAISS
    pairs = extract_text_from_pdfs(paths)
    new_docs, chunk_rows = chunk_text(
        pairs,
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
    )
    if not new_docs:
        raise HTTPException(status_code=400, detail="No extractable text found in the uploaded PDFs.")
    if chunk_rows:
        save_chunks(chunk_rows, db_path=DB_PATH)
    all_docs = load_documents_from_db(DB_PATH)
    if not all_docs:
        raise HTTPException(status_code=500, detail="Failed to persist chunks to the document store.")
    vs = build_or_load_faiss(all_docs, persist_dir=PERSIST_DIR)

    # Refresh global chain
    global _chain
    _chain = make_conversation_chain(vs)
    return {"status": "ok", "documents": len(all_docs), "ingested": len(new_docs)}

@app.post("/ask")
async def ask(body: AskBody):
    chain = _ensure_chain()
    result = chain.invoke({"question": body.question})
    answer = result["answer"]
    sources = [{"source": d.metadata.get("source", ""), "snippet": d.page_content[:200]} for d in result["source_documents"]]
    return {"answer": answer, "sources": sources}

# ---------- CLI Driver (optional) ----------
def cli():
    print("CLI Q&A. Type 'exit' to quit.")
    # Try to auto-load vector store
    from langchain_community.vectorstores import FAISS
    try:
        vs = FAISS.load_local(PERSIST_DIR, embeddings=None, allow_dangerous_deserialization=True)
        from .vectorstore import _get_embeddings
        vs.embeddings = _get_embeddings()
    except Exception:
        print("No vectorstore found. Please run the /ingest endpoint or call ingest_cli() first.")
        return
    chain = make_conversation_chain(vs)
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}: break
        res = chain.invoke({"question": q})
        print("\nBot:", res["answer"])

if __name__ == "__main__":
    # Run: uvicorn backend.app:app --reload
    cli()



# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     # embeddings = HuggingFaceEmbeddings(
#     #     model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     # llm = HuggingFacePipeline.from_model_id(
#     #     model_id="lmsys/vicuna-7b-v1.3",
#     #     task="text-generation",
#     #     model_kwargs={"temperature": 0.01},
#     # )
#     # llm = LlamaCpp(
#     #     model_path="models/llama-2-7b-chat.ggmlv3.q4_1.bin",  n_ctx=1024, n_batch=512)

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(
#             search_type="similarity", search_kwargs={"k": 4}),
#         memory=memory,
#     )
#     return conversation_chain


# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with PDFs",
#                        page_icon=":robot_face:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with PDFs :robot_face:")
#     user_question = st.text_input("Ask questions about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get pdf text
#                 raw_text = get_pdf_text(pdf_docs)

#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(
#                     vectorstore)


# NOTE: main() is not defined in this module. To avoid running on import,
# comment out the call below. If you add a main() later, you can uncomment.
# if __name__ == '__main__':
#     main()
