import os
from typing import List

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def _get_llm():
    provider = os.getenv("PROVIDER", "openai").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMP", "0.0"))
        )
    else:
        # Open-source local LLM via Ollama (default to llama3)
        model = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = (
            os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://host.docker.internal:11434"
        )
        return ChatOllama(
            model=model,
            temperature=float(os.getenv("TEMP", "0.0")),
            base_url=base_url,
        )

def _format_docs(docs: List):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        tag = f"[{source}{f' p.{page}' if page else ''}]"
        formatted.append(f"{tag} {doc.page_content}")
    return "\n\n".join(formatted)

class ConversationEngine:
    def __init__(self, retriever, prompt):
        self.retriever = retriever
        self.prompt = prompt
        self.llm = _get_llm()
        self.chat_history: List = []

    def invoke(self, inputs):
        question = inputs.get("question", "")
        docs = self.retriever.invoke(question)
        context = _format_docs(docs)
        messages = self.prompt.format_messages(
            chat_history=self.chat_history,
            question=question,
            context=context
        )
        response = self.llm.invoke(messages)
        if not isinstance(response, AIMessage):
            response_msg = AIMessage(content=getattr(response, "content", str(response)))
        else:
            response_msg = response

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(response_msg)

        return {"answer": response_msg.content, "source_documents": docs}

def make_conversation_chain(vectorstore, k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions using the provided context."),
        MessagesPlaceholder("chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    return ConversationEngine(retriever, prompt)
