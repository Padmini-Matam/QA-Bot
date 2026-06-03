# QueryVault — Decision Structure

This document explains **every design decision** in plain English: what was chosen, why, and what the alternatives were.

---

## 1. Overall Architecture: Why RAG?

**Decision:** Use Retrieval-Augmented Generation (RAG) instead of fine-tuning an LLM on documents.

**Reasoning:**
- Fine-tuning requires thousands of examples, a GPU, and hours of training. RAG requires none of that.
- RAG works with *any* document at runtime — you don't retrain the model for each new PDF.
- RAG answers are grounded in actual document text, reducing hallucinations.
- Fine-tuned models "bake in" knowledge and can't be updated without retraining. RAG just needs a new document upload.

**Alternative considered:** Prompt-stuffing (paste the whole PDF into the LLM prompt).
**Why rejected:** LLMs have a context window limit (~32K tokens for Mixtral). A 100-page PDF has ~75K+ tokens. It won't fit and would be very expensive even if it did.

---

## 2. Document Loader: PyPDFLoader

**Decision:** Use LangChain's `PyPDFLoader` to parse PDFs.

**Reasoning:**
- Returns one `Document` object per page, making it easy to track page numbers.
- Handles text extraction from multi-page PDFs without manual file I/O.
- Integrates natively with the rest of the LangChain pipeline.

**Alternative considered:** `PDFMinerLoader` or `pdfplumber`.
**Why rejected:** PyPDFLoader is simpler and sufficient for text-based PDFs. PDFMiner is better for complex layouts (tables, columns) but adds complexity.

**Limitation:** Scanned PDFs (images of text) won't work without an OCR layer (e.g., Tesseract). This is noted as a future enhancement.

---

## 3. Text Splitter: RecursiveCharacterTextSplitter

**Decision:** Use `RecursiveCharacterTextSplitter` with chunk_size=1000, chunk_overlap=100.

**Reasoning:**
- A raw page can be 3000–5000 characters — too long for an embedding model to represent well as a single vector.
- Smaller chunks = more precise retrieval (the retrieved chunk is more likely to contain only the relevant answer).
- `RecursiveCharacterTextSplitter` is smarter than a fixed character splitter: it tries to split at paragraph breaks, then sentence breaks, then word breaks — so chunks rarely cut mid-sentence.
- `chunk_overlap=100` means the last 100 characters of one chunk repeat as the first 100 of the next. This prevents answers that span chunk boundaries from being missed.

**Why chunk_size=1000:**
- Watsonx Slate 125M embedding model handles up to ~512 tokens (~1500 chars).
- 1000 characters ≈ 250 tokens — well within limit, fast to embed, precise to retrieve.

**Alternative considered:** `CharacterTextSplitter` (splits only on `\n`).
**Why rejected:** Produces uneven chunk sizes; a single paragraph can be 5000 characters, breaking the embedding model's limit.

---

## 4. Embedding Model: IBM Slate 125M English

**Decision:** Use `WatsonxEmbeddings` with `IBM_SLATE_125M_ENG`.

**Reasoning:**
- Available free under the IBM Watsonx free tier — no additional cost.
- 125M parameters is enough to capture semantic meaning for retrieval tasks.
- Produces 768-dimensional vectors — a good balance of expressiveness and storage efficiency.
- Using the same IBM platform for both embeddings and LLM keeps credentials simple (one API key, one project ID).

**Alternative considered:** OpenAI `text-embedding-ada-002`.
**Why rejected:** Requires a separate OpenAI API key and charges per token. IBM Watsonx is already being used for the LLM, so keeping the same platform reduces complexity.

**What embeddings actually do:**
Each text chunk is converted to a list of ~768 numbers. Chunks with similar meaning produce number lists that are close together in 768-dimensional space. This is what makes semantic search possible — "What is Hadoop?" finds chunks about Hadoop installation even if those chunks never use the word "what".

---

## 5. Vector Database: ChromaDB

**Decision:** Store embeddings in ChromaDB, persisted to `./chroma_db/` on disk.

**Reasoning:**
- ChromaDB is a local vector database — no server setup, no Docker, no cloud account.
- Persisting to disk means re-uploading the same PDF doesn't re-embed it (future optimization possibility).
- Has a simple Python API that LangChain wraps natively.
- Fast enough for documents up to ~500 pages in local use.

**Alternative considered:** Pinecone (cloud), FAISS (in-memory).
**Why rejected:**
- Pinecone requires a cloud account and has latency.
- FAISS is in-memory only — embeddings are lost when the app restarts.
- ChromaDB gives the best balance of simplicity, persistence, and local control for this use case.

**How retrieval works:**
When a user asks a question, it's also embedded into a vector. ChromaDB then finds the top-k (k=3) stored chunk vectors that are closest to the question vector using **cosine similarity**. These are the "most relevant" chunks.

---

## 6. LLM: Mixtral 8×7B Instruct

**Decision:** Use `mistralai/mixtral-8x7b-instruct-v01` via IBM Watsonx.

**Reasoning:**
- **Mixture of Experts (MoE) architecture**: Mixtral has 8 expert sub-networks but only activates 2 per token. This gives ~46B parameter quality at ~13B parameter inference cost.
- "Instruct" variant is fine-tuned to follow instructions — critical for QA where the model must answer a specific question using only provided context.
- Available through IBM Watsonx without managing GPU infrastructure.

**Generation parameters chosen:**
- `DECODING_METHOD: GREEDY` — always picks the highest-probability next token. Produces deterministic, factual responses (good for QA). Sampling would introduce randomness appropriate for creative tasks but not for document Q&A.
- `MAX_NEW_TOKENS: 500` — enough for a thorough answer without runaway generation.
- `STOP_SEQUENCES: ["<|endoftext|>"]` — prevents the model from generating beyond its natural stopping point.

**Alternative considered:** IBM Granite, Llama 3.
**Why Mixtral:** Best instruction-following and reasoning performance available on Watsonx at time of implementation.

---

## 7. QA Chain: RetrievalQA with "stuff" chain type

**Decision:** Use LangChain's `RetrievalQA` with `chain_type="stuff"`.

**What "stuff" means:** All retrieved chunks are "stuffed" (concatenated) into a single prompt and sent to the LLM at once, alongside the question.

**Reasoning:**
- With k=3 chunks of ~1000 characters each, the total context is ~3000 characters — well within Mixtral's 32K token window.
- Simple, fast, and produces coherent answers because the LLM sees all context in one pass.
- `return_source_documents=True` lets you optionally show users which pages the answer came from.

**Alternative chain types:**
- `map_reduce` — summarizes each chunk separately, then summarizes the summaries. Good for very long documents but slower and loses cross-chunk context.
- `refine` — iteratively refines an answer chunk by chunk. Better quality for complex synthesis but ~3× slower.
- `map_rerank` — scores each chunk independently and picks the best answer. More precise but doesn't synthesize across chunks.

**Why "stuff" wins here:** For a QA bot with small k and moderate chunk sizes, "stuff" is fastest and most coherent.

---

## 8. Gradio Interface

**Decision:** Use Gradio `Blocks` layout with a file upload + textbox + button.

**Reasoning:**
- `gr.Blocks` gives layout control (two-column design) vs. `gr.Interface` which is single-column only.
- `gr.File(type="filepath")` returns the temporary file path that `PyPDFLoader` needs.
- `gr.themes.Soft()` provides a clean, professional look without custom CSS.
- Gradio handles file upload, session state, and server/browser communication automatically.

**Why not Flask/FastAPI + custom HTML:**
- Would require writing frontend HTML/CSS/JS.
- Gradio gives a production-quality UI in ~30 lines.
- Fine for a portfolio project and can be replaced later if needed.

---

## 9. Credential Management: Environment Variables + .env

**Decision:** Read API keys from environment variables; provide `.env.example` but gitignore `.env`.

**Reasoning:**
- Hardcoding API keys in source code is a critical security mistake — GitHub scans for leaked keys and IBM/cloud providers rotate them automatically when detected.
- `python-dotenv` loads `.env` into `os.environ` at runtime.
- `.env.example` shows collaborators what credentials are needed without exposing real values.

---

## 10. File and Module Structure

```
app.py          — entry point; only UI code
src/qabot.py    — all business logic; imported by app.py
```

**Reasoning:**
- Separation of concerns: UI layer never contains business logic.
- `qabot.py` can be tested, imported, or replaced independently of the UI.
- If you later add a FastAPI backend, you import `process_pdf_and_query` from `qabot.py` without touching `app.py`.
