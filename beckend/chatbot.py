import os
import logging
from typing import List

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)

def _get_llm():
    provider = os.getenv("PROVIDER", "openai").lower()
    logger.info(f"Initializing LLM with provider: {provider}")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        logger.info(f"Using OpenAI model: {model}")
        return ChatOpenAI(
            model=model,
            temperature=float(os.getenv("TEMP", "0.0"))
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        logger.info(f"Using Gemini model: {model}")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=float(os.getenv("TEMP", "0.0")),
            google_api_key=api_key
        )
    else:
        # Open-source local LLM via Ollama (default to llama3)
        model = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = (
            os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://host.docker.internal:11434"
        )
        logger.info(f"Using Ollama model: {model} at {base_url}")
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
        logger.info(f"Processing question: {question[:100]}...")

        try:
            # Retrieve relevant documents
            logger.debug(f"Retrieving documents for question...")
            docs = self.retriever.invoke(question)
            logger.info(f"Retrieved {len(docs)} documents")

            # Format context from documents
            context = _format_docs(docs)
            logger.debug(f"Context length: {len(context)} characters")

            # Format the prompt with chat history
            logger.debug(f"Formatting prompt with {len(self.chat_history)} previous messages")
            messages = self.prompt.format_messages(
                chat_history=self.chat_history,
                question=question,
                context=context
            )
            logger.debug(f"Formatted {len(messages)} messages for LLM")

            # Invoke LLM
            logger.info("Invoking LLM...")
            response = self.llm.invoke(messages)
            logger.info("LLM response received")

            if not isinstance(response, AIMessage):
                response_msg = AIMessage(content=getattr(response, "content", str(response)))
            else:
                response_msg = response

            # Update chat history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(response_msg)
            logger.debug(f"Chat history updated. Total messages: {len(self.chat_history)}")

            return {"answer": response_msg.content, "source_documents": docs}

        except Exception as e:
            logger.error(f"Error in conversation engine: {str(e)}", exc_info=True)
            raise

def make_conversation_chain(vectorstore, k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions using the provided context."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    return ConversationEngine(retriever, prompt)
