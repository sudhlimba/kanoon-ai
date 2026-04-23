"""
rag_pipeline.py
----------------
Core RAG pipeline for KanoonAI — Indian Legal Document Analyzer.

Pipeline steps:
  1. Load multiple PDF documents (all 4 acts at once)
  2. Split into overlapping chunks
  3. Embed with OpenAI and store in ChromaDB
  4. On query: retrieve top-K relevant chunks
  5. Pass to GPT with a bilingual (Hindi+English) citizen-friendly prompt
  6. Return answer + sources

Key design decisions:
  - Single shared ChromaDB collection for ALL documents
    → user can ask cross-document questions ("How does RTI relate to IT Act?")
  - Each chunk keeps metadata: source filename + page number
    → shown to user as "Source: RTI Act 2005 · Page 4"
  - Custom prompt forces simple, jargon-free language
  - Hindi detection: if question contains Devanagari script → answer in Hindi
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_DB_DIR        = "chroma_db"
COLLECTION_NAME      = "indian_legal_docs"
CHUNK_SIZE           = 1200       # slightly larger for legal text (long sentences)
CHUNK_OVERLAP        = 250
EMBEDDING_MODEL      = "text-embedding-3-small"
CHAT_MODEL           = "gpt-4o-mini"
TOP_K_RESULTS        = 5          # retrieve 5 chunks (legal answers need more context)

# ── Document display names (filename → readable name) ─────────────────────────
DOC_DISPLAY_NAMES = {
    "indian_constitution.pdf"        : "Indian Constitution",
    "rti_act_2005.pdf"               : "RTI Act 2005",
    "it_act_2000.pdf"                : "IT Act 2000",
    "consumer_protection_act_2019.pdf": "Consumer Protection Act 2019",
}

# ── Bilingual Citizen-Friendly Prompt ─────────────────────────────────────────
#
# Key instructions inside the prompt:
#   1. Answer in Hindi if the question is in Hindi, else English
#   2. Use simple language — no legal jargon without explanation
#   3. If information not found, say so honestly (no hallucination)
#   4. Always mention which law/act the answer comes from
#
PROMPT_TEMPLATE = """
You are KanoonAI, a helpful Indian legal assistant for common people.

Your goal:
Explain Indian laws in a simple, clear, and practical way.

RULES:
1. Use the provided context as your primary source.
2. If the context is not enough, you may give a general explanation BUT clearly say it's general knowledge.
3. Answer in the SAME language as the user (Hindi or English).
4. Keep explanations simple and easy — like explaining to a beginner.
5. Always mention the law/act when possible.
6. Keep answers SHORT (maximum 4–5 lines).
7. Use bullet points ONLY if necessary.
8. Do NOT over-explain unless the user asks for more details.
9. Avoid unnecessary background information.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=PROMPT_TEMPLATE,
)


# ── Step 1 & 2: Load and chunk all PDFs ───────────────────────────────────────

def load_and_split_documents(pdf_paths: list) -> list:
    """
    Load multiple PDF files and split them into chunks.
    Each chunk's metadata includes: source filename, page number, document name.

    Args:
        pdf_paths: List of full file paths to PDFs.

    Returns:
        Combined list of Document chunks from all PDFs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        length_function = len,
        separators      = ["\n\n", "\n", ".", " ", ""],  # legal docs have lots of \n
    )

    all_chunks = []

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"  📄 Loading: {filename}")

        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()

        # Add readable document name to every page's metadata
        for page in pages:
            page.metadata["document_name"] = DOC_DISPLAY_NAMES.get(filename, filename)
            page.metadata["filename"]      = filename

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"     → {len(pages)} pages → {len(chunks)} chunks")

    print(f"\n  Total chunks across all documents: {len(all_chunks)}")
    return all_chunks


# ── Step 3: Build ChromaDB vector store ───────────────────────────────────────

def build_vector_store(chunks: list) -> Chroma:
    """
    Embed all chunks and store them in ChromaDB.
    This is the expensive step — calls OpenAI Embeddings API for each chunk.

    Args:
        chunks: All document chunks from load_and_split_documents().

    Returns:
        Chroma vector store object.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    print(f"  🔢 Embedding {len(chunks)} chunks via OpenAI API...")
    print("     (This may take 1-3 minutes for the first run)")

    vector_store = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        collection_name   = COLLECTION_NAME,
        persist_directory = CHROMA_DB_DIR,
    )
    print("  ✅ Vector store saved to disk (chroma_db/)")
    return vector_store


def load_existing_vector_store() -> Chroma:
    """
    Load already-built vector store from disk.
    Used on subsequent app starts — no re-embedding needed.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name    = COLLECTION_NAME,
        embedding_function = embeddings,
        persist_directory  = CHROMA_DB_DIR,
    )


def vector_store_exists() -> bool:
    """Check if a ChromaDB collection already exists on disk."""
    chroma_dir = os.path.join(CHROMA_DB_DIR)
    return os.path.exists(chroma_dir) and len(os.listdir(chroma_dir)) > 0


# ── Step 4 & 5: Build the full RAG chain ─────────────────────────────────────

def build_rag_chain(vector_store: Chroma) -> ConversationalRetrievalChain:
    """
    Assemble the complete RAG chain:
        Retriever → Memory → LLM → Answer

    Args:
        vector_store: Chroma vector store with all embedded legal doc chunks.

    Returns:
        ConversationalRetrievalChain ready to answer questions.
    """
    llm = ChatOpenAI(
        model_name  = CHAT_MODEL,
        temperature = 0.3,    # very low: legal answers must be factual, not creative
    )

    # MMR (Maximal Marginal Relevance) = retrieves diverse chunks, not just the most similar
    # This is better for legal text where the same section may repeat
    retriever = vector_store.as_retriever(
        search_type   = "mmr",
        search_kwargs = {
            "k"               : TOP_K_RESULTS,
            "fetch_k"         : 10,   # fetch 10, then pick 5 most diverse
            "lambda_mult"     : 0.7,  # 0=max diversity, 1=max similarity
        },
    )

    memory = ConversationBufferWindowMemory(
     k=5,
     memory_key="chat_history",
     return_messages=True,
     output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm                       = llm,
        retriever                 = retriever,
        memory                    = memory,
        combine_docs_chain_kwargs = {"prompt": CUSTOM_PROMPT},
        return_source_documents   = True,
        verbose                   = False,
    )
    return chain


# ── Main entry point ──────────────────────────────────────────────────────────

def initialize_rag_system(pdf_paths: list) -> ConversationalRetrievalChain:
    """
    Full initialization:
      - If vector store already on disk → load it (fast, no API calls)
      - Else → build from scratch (slow, calls OpenAI Embeddings)

    Args:
        pdf_paths: Paths to all PDF documents.

    Returns:
        Ready-to-use ConversationalRetrievalChain.
    """
    if vector_store_exists():
        print("\n⚡ Found existing vector store. Loading from disk (fast)...")
        vector_store = load_existing_vector_store()
    else:
        print("\n🔨 Building vector store for the first time...")
        chunks       = load_and_split_documents(pdf_paths)
        vector_store = build_vector_store(chunks)

    chain = build_rag_chain(vector_store)
    print("✅ KanoonAI is ready!\n")
    return chain


def ask_question(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Ask a legal question using the RAG chain.

    Args:
        chain:    The initialized ConversationalRetrievalChain.
        question: User's question (Hindi or English).

    Returns:
        Dict with:
          'answer'           → str  (GPT's answer)
          'source_documents' → list (chunks used to generate the answer)
    """
    result = chain.invoke({"question": question})
    return {
        "answer"          : result["answer"],
        "source_documents": result.get("source_documents", []),
    }


def rebuild_vector_store(pdf_paths: list) -> ConversationalRetrievalChain:
    """
    Force-rebuild the vector store (deletes existing one first).
    Used when user wants to re-index after adding a new document.
    """
    import shutil
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        print("🗑️  Deleted old vector store.")

    chunks       = load_and_split_documents(pdf_paths)
    vector_store = build_vector_store(chunks)
    return build_rag_chain(vector_store)
