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
from langchain_ibm import ChatWatsonx, WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, EmbeddingTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods


# ---------------------------------------------------------------------------
# 1. LLM Initialization
# ---------------------------------------------------------------------------

def init_llm() -> ChatWatsonx:
    """
    Initialize the IBM Watsonx LLM (Mixtral 8x7B Instruct).

    Reads credentials from environment variables:
        WATSONX_API_KEY  – your IBM Cloud API key
        WATSONX_PROJECT_ID – your Watsonx project ID
        WATSONX_URL      – IBM Cloud endpoint (default provided)

    Returns:
        ChatWatsonx instance ready for inference.
    """
    parameters = {
        "max_new_tokens": 500,
        "min_new_tokens": 1,
    }

    llm = ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        apikey=os.getenv("WATSONX_API_KEY"),
        params=parameters,
    )
    return llm


# ---------------------------------------------------------------------------
# 2. Embedding Model
# ---------------------------------------------------------------------------

def watsonx_embedding() -> WatsonxEmbeddings:
    """
    Initialize IBM Watsonx Embeddings (Slate 125M English).

    Returns:
        WatsonxEmbeddings instance.
    """
    embeddings = WatsonxEmbeddings(
        model_id="ibm/slate-30m-english-rtrvr-v2",
        url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        apikey=os.getenv("WATSONX_API_KEY"),
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
    embedding_model = watsonx_embedding()
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

def build_qa_chain(retriever, llm: ChatWatsonx):
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
    if not question.strip():
        return "Please enter a question."

    try:
        # Step 1 – Load
        documents = load_pdf(file_path)

        # Step 2 – Split
        chunks = split_documents(documents)

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
