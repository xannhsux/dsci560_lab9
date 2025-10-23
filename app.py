from csv import reader
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain.memory import ConversationBufferMemory
#from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
#from langchain_community.llms import LlamaCpp
import sqlite3, time


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


def save_chunks(chunks, db_path="ads_texts.db"):
   
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS pdf_chunks")
    cur.execute("""CREATE TABLE pdf_chunks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        page INTEGER,
        chunk_id INTEGER,
        chunk_text TEXT,
        created_at INTEGER
    )""")

    now = int(time.time())
    rows = []
    for chunk in chunks:
  
        try:
            if len(chunk) == 4:
                f, p, i, t = chunk
            elif len(chunk) == 3:
                f, i, t = chunk
                p = None
            else:
  
                continue
            rows.append((f, p, i, t, now))
        except Exception:
            continue

    if rows:
        cur.executemany(
            "INSERT INTO pdf_chunks(filename,page,chunk_id,chunk_text,created_at) VALUES (?,?,?,?,?)",
            rows,
        )
    conn.commit()
    conn.close()



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
