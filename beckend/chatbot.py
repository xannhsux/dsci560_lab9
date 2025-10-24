import os
from langchain_community.chat_models import ChatOllama
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

def _get_llm():
    provider = os.getenv("PROVIDER", "openai").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMP", "0.0"))
        )
    else:
        # Open-source local LLM via Ollama (free). Example: `ollama run llama3.1`
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        return ChatOllama(model=model, temperature=float(os.getenv("TEMP", "0.0")))

class ConversationEngine:
    def __init__(self, chain):
        self.chain = chain
        self.chat_history = []

    def invoke(self, inputs):
        question = inputs.get("question")
        payload = {
            "question": question,
            "chat_history": self.chat_history,
        }
        result = self.chain.invoke(payload)
        self.chat_history.append((question, result.get("answer", "")))
        return result

def make_conversation_chain(vectorstore, k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chain = ConversationalRetrievalChain.from_llm(
        llm=_get_llm(),
        retriever=retriever,
        return_source_documents=True
    )
    return ConversationEngine(chain)
