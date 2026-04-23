# ⚖️ KanoonAI — Indian Legal Document Analyzer

A **RAG-powered AI assistant** that answers questions about Indian law in simple Hindi and English — built for common citizens, not lawyers.

Powered by **LangChain + OpenAI + ChromaDB + Streamlit**.

---

## 📚 Preloaded Legal Documents

| Document | Source |
|---|---|
| 🇮🇳 Indian Constitution | legislative.gov.in |
| 📋 RTI Act 2005 | rti.gov.in |
| 💻 IT Act 2000 | legislative.gov.in |
| 🛒 Consumer Protection Act 2019 | consumeraffairs.nic.in |

---

## 🧠 How RAG Works Here

```
4 Legal PDFs
     ↓
[PyPDFLoader] — load all pages from each PDF
     ↓
[RecursiveCharacterTextSplitter] — break into 1200-char overlapping chunks
     ↓
[OpenAI text-embedding-3-small] — convert each chunk to a vector
     ↓
[ChromaDB] — store all vectors locally on disk
     ↓
User asks: "What are my consumer rights?"
     ↓
[MMR Similarity Search] — retrieve top 5 most relevant chunks
     ↓
[GPT-3.5-Turbo + Custom Prompt] — generate citizen-friendly answer
     ↓
Answer shown in Hindi or English (auto-detected) + source references
```

---

## 📁 Project Structure

```
kanoon_ai/
│
├── app.py                    ← Streamlit UI
├── rag_pipeline.py           ← Core RAG logic
├── document_downloader.py    ← Downloads PDFs automatically
├── requirements.txt
├── .env                      ← Your OpenAI key (never commit!)
├── .gitignore
├── README.md
│
├── documents/                ← Auto-created, holds the 4 PDFs
└── chroma_db/                ← Auto-created, holds the vector store
```

---

## 🛠️ Setup on Windows (copy-paste ready)

### Step 1 — Create virtual environment
```
cd kanoon_ai
python -m venv venv
venv\Scripts\activate
```

### Step 2 — Install dependencies
```
pip install -r requirements.txt
```

### Step 3 — Add your OpenAI API key
Open `.env` and set:
```
OPENAI_API_KEY=sk-your-key-here
```

### Step 4 — Download the legal PDFs
```
python document_downloader.py
```
This creates a `documents/` folder and downloads all 4 PDFs automatically.
> If auto-download fails for any PDF, the script tells you the manual download link.

### Step 5 — Run the app
```
streamlit run app.py
```
Opens at `http://localhost:8501`

### Step 6 — Initialize (first time only)
- Enter your API key in the sidebar
- Click **Initialize KanoonAI**
- Wait ~2 minutes while documents are embedded (only done once!)
- Next time you start the app, it loads instantly from disk

---

## ☁️ Deploy on Streamlit Cloud

1. Push to GitHub (`.env` and `documents/` and `chroma_db/` are in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. In **Advanced Settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
4. **Important for Streamlit Cloud**: Since PDFs can't be pre-uploaded, modify `document_downloader.py` to download them at startup, OR add the PDFs directly to your GitHub repo (they are public domain government documents, so this is fine).

---

## ✨ Features

- ✅ 4 preloaded Indian legal acts — no uploading needed
- ✅ Ask in **Hindi or English** — auto-detected
- ✅ **8 suggested questions** with one-click buttons
- ✅ Shows **exactly which document and page** the answer came from
- ✅ **Cross-document questions** — "How does RTI relate to the Constitution?"
- ✅ **Conversation memory** — ask follow-up questions naturally
- ✅ **MMR retrieval** — diverse, non-redundant chunks
- ✅ Anti-hallucination prompt — says "not found" instead of making things up
- ✅ Citizen-friendly language — no legal jargon

---

## 💼 Resume Description

> **KanoonAI – Indian Legal Document Analyzer** | Python, LangChain, OpenAI, ChromaDB, Streamlit
>
> Built a bilingual (Hindi/English) RAG application enabling citizens to query 4 Indian legal acts (Constitution, RTI Act, IT Act, Consumer Protection Act). Implemented multi-document ingestion with metadata tagging, MMR-based semantic retrieval via ChromaDB, and GPT-3.5-Turbo with a custom anti-hallucination prompt. Features conversation memory, auto language detection, source attribution with page references, and one-click suggested questions. Deployed on Streamlit Cloud.

---

## 🙋 Sample Questions to Try

**English:**
- What are my Fundamental Rights?
- How do I file an RTI application? What is the time limit?
- What can I do if an e-commerce company cheats me?
- Is hacking someone's account a crime? What is the punishment?
- Can the government tap my phone?

**Hindi:**
- मेरे मौलिक अधिकार क्या हैं?
- RTI कैसे दाखिल करें?
- अगर कोई कंपनी मुझे धोखा दे तो मैं क्या कर सकता हूँ?
