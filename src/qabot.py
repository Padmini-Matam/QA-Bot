"""
QueryVault - PDF Question Answering Bot
Core backend logic: document loading, embedding, retrieval, and LLM response generation.
"""

import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# 1. LLM Initialization
# ---------------------------------------------------------------------------

def init_llm() -> ChatGroq:
    """
    Initialize the Groq LLM (Llama 3 70B Instruct).

    Reads credentials from environment variables:
        GROQ_API_KEY  – your Groq API key

    Returns:
        ChatGroq instance ready for inference.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )
    return llm


# ---------------------------------------------------------------------------
# 2. Embedding Model
# ---------------------------------------------------------------------------

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace Local Embeddings (all-MiniLM-L6-v2).

    Returns:
        HuggingFaceEmbeddings instance.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return embeddings


# ---------------------------------------------------------------------------
# 3. Document Loader
# ---------------------------------------------------------------------------

def load_pdf(file_path: str) -> list:
    """
    Load and parse a PDF file using PyPDFLoader.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of LangChain Document objects (one per page).
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# ---------------------------------------------------------------------------
# 4. Text Splitter
# ---------------------------------------------------------------------------

def split_documents(documents: list) -> list:
    """
    Split loaded documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter with:
        chunk_size  = 1000 characters
        chunk_overlap = 100 characters (preserves boundary context)

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of smaller Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ---------------------------------------------------------------------------
# 5. Vector Store
# ---------------------------------------------------------------------------

def create_vector_store(chunks: list) -> Chroma:
    """
    Convert document chunks into embeddings and store in ChromaDB.

    Args:
        chunks: List of Document chunks from the text splitter.

    Returns:
        Chroma vector store instance.
    """
    embedding_model = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=f"pdf_{uuid.uuid4().hex}"
    )
    return vector_store


# ---------------------------------------------------------------------------
# 6. Retriever
# ---------------------------------------------------------------------------

def get_retriever(vector_store: Chroma):
    """
    Create a retriever from the vector store using similarity search.

    Args:
        vector_store: Chroma vector store instance.

    Returns:
        LangChain BaseRetriever configured for top-3 similarity search.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )
    return retriever


# ---------------------------------------------------------------------------
# 7. QA Chain
# ---------------------------------------------------------------------------

def build_qa_chain(retriever, llm: ChatGroq):
    """
    Build a QA chain combining the retriever and LLM using LCEL.

    Args:
        retriever: LangChain retriever from vector store.
        llm: Initialized ChatWatsonx instance.

    Returns:
        Runnable chain ready to answer questions.
    """
    template = """You are a helpful AI assistant answering questions based on a provided document.
Use the following pieces of context from the document to answer the question. If the question is general (like "what is this document about"), summarize the provided context. If the context does not contain the answer, say "I don't have enough information from the document to answer that."

Context:
{context}

Question: {question}

Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain


# ---------------------------------------------------------------------------
# 8. Full Pipeline: process PDF and answer a question
# ---------------------------------------------------------------------------

def process_pdf_and_query(file_path: str, question: str) -> str:
    """
    End-to-end pipeline: load PDF → split → embed → retrieve → answer.

    Args:
        file_path: Path to the uploaded PDF file.
        question:  User's natural language question.

    Returns:
        String answer generated by the LLM.
    """
    if not file_path:
        return "Please upload a PDF file first."
    if not question or not str(question).strip():
        return "Please enter a question."

    try:
        # Step 1 – Load
        documents = load_pdf(file_path)

        # Step 2 – Split
        chunks = split_documents(documents)
        if not chunks:
            return "Error: Could not extract any readable text from the PDF. It might be a scanned image or password-protected."

        # Step 3 & 4 – Embed + Store
        vector_store = create_vector_store(chunks)

        # Step 5 – Retrieve
        retriever = get_retriever(vector_store)

        # Step 6 – LLM
        llm = init_llm()

        # Step 7 – QA Chain
        qa_chain = build_qa_chain(retriever, llm)

        # Step 8 – Run
        answer = qa_chain.invoke(question)
        if not answer:
            answer = "No answer could be generated."
        return answer

    except Exception as e:
        return f"Error processing your request: {str(e)}"
